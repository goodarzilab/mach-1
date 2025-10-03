#!/usr/bin/env python

import os
import argparse
import yaml
import math
from typing import List

import torch
from torch.nn import Embedding, LayerNorm

from stripedhyena.utils import dotdict
from stripedhyena.layers import RMSNorm
from configuration_hyena import StripedHyenaConfig
from modeling_hyena import StripedHyenaModelForCausalLM

from datasets import load_from_disk
from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback)
from transformers.trainer_pt_utils import get_parameter_names

import wandb

no_decay_layer_types = (Embedding, LayerNorm, RMSNorm)
no_decay_layer_names = ['bias', 'poles', 'residues']

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trainer_config', type=str, help='Path to trainer config file')
parser.add_argument('-m', '--model_config', type=str, help='Path to model config file')
parser.add_argument('-p', '--num_threads', type=int, help='Number of threads', default=16)

args = parser.parse_args()

with open(args.trainer_config, 'r') as yaml_file:
    trainer_config_yaml = yaml.safe_load(yaml_file)
    config = dotdict(trainer_config_yaml)

with open(args.model_config, 'r') as yaml_file:
    model_config_yaml = yaml.safe_load(yaml_file)
    model_config = dotdict(model_config_yaml)

set_seed(config.seed)
transformers_set_seed(config.seed)

project_output_dir = f"{config.output_dir}/{config.wandb_project}"
os.makedirs(project_output_dir, exist_ok=True)

my_model_config = StripedHyenaConfig(**model_config)
model = StripedHyenaModelForCausalLM(my_model_config)

if config.from_pretrained:
    model = StripedHyenaModelForCausalLM.from_pretrained(config.from_pretrained, config=my_model_config)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=config.tokenizer,
    padding_side='right',
    truncation_side='right',
    cls_token='[CLS]',
    bos_token='[CLS]',
    sep_token='[SEP]',
    eos_token='[SEP]',
    unk_token='[UNK]',
    mask_token='[MASK]',
    pad_token='[PAD]',
    model_max_length=model_config.max_seqlen)

tokenized_datasets = load_from_disk(config.tokenized_dataset)

dataset_columns = tokenized_datasets.column_names['train']
columns_to_keep = ['input_ids', 'attention_mask', 'special_tokens_mask', 'seq_len']
columns_to_keep = list(set(dataset_columns) & set(columns_to_keep))
tokenized_datasets = tokenized_datasets.select_columns(columns_to_keep)

num_train_samples = len(tokenized_datasets['train'])
num_eval_samples = len(tokenized_datasets['validation'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)

model_name = f"RNA_V{model_config.vocab_size}_H{model_config.hidden_size}_N{model_config.num_layers}"
if config.trainer_id:
    model_name += f"_{config.trainer_id}"

wandb.init(project = config.wandb_project,
           config = {**config, **model_config},
           allow_val_change = True,
           name = model_name,
           resume = 'allow',
           save_code = True)

### Estimate batch size
def find_max_batch_size(model, tokenizer, max_length, initial_batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    low, high = initial_batch_size, initial_batch_size * 32
    max_batch_size = initial_batch_size

    while low <= high:
        mid = (low + high) // 2
        try:
            sample_input = torch.randint(0, tokenizer.vocab_size, (mid, max_length))
            sample_input = sample_input.to(device)

            with torch.no_grad():
                _ = model(sample_input)

            max_batch_size = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
            else:
                raise e

    return max_batch_size

max_batch_size = find_max_batch_size(model, tokenizer, model_config.max_seqlen)
steps_per_train_epoch = num_train_samples // max_batch_size
steps_per_eval_loop = num_eval_samples // max_batch_size

training_args = TrainingArguments(
    seed = config.seed,
    output_dir=f"{project_output_dir}/{model_name}",
    bf16=True,
    save_safetensors=True,
    report_to='wandb',
    disable_tqdm = False,
    run_name=model_name,
    num_train_epochs = config.num_train_epochs,
    per_device_train_batch_size = config.batch_size if config.batch_size else max_batch_size,
    per_device_eval_batch_size = config.batch_size if config.batch_size else max_batch_size,
    gradient_accumulation_steps = config.gradient_accumulation_steps if config.gradient_accumulation_steps else 1,
    eval_accumulation_steps = config.eval_accumulation_steps if config.eval_accumulation_steps else 128,
    logging_strategy = 'steps',
    logging_first_step = True,
    logging_steps = config.logging_steps if config.logging_steps else int(steps_per_train_epoch * 0.025),
    evaluation_strategy = 'steps',
    eval_steps = config.eval_steps if config.eval_steps else int(steps_per_eval_loop * 2),
    eval_delay = 0,
    save_strategy = 'steps',
    save_steps = config.save_steps if config.save_steps else int(steps_per_eval_loop * 2),
    warmup_ratio = config.warmup_ratio,
    learning_rate = config.learning_rate,
    lr_scheduler_type = config.lr_scheduler_type,
    group_by_length = config.group_by_length,
    length_column_name = config.length_column_name,
    auto_find_batch_size = config.group_by_length,
    dataloader_num_workers = args.num_threads,
    dataloader_persistent_workers = True,
    max_grad_norm = config.max_grad_norm if config.max_grad_norm else 1.0,
    weight_decay = config.weight_decay if config.weight_decay else 0.0,
    load_best_model_at_end = config.load_best_model_at_end,
    metric_for_best_model = config.metric_for_best_model,
    greater_is_better = config.greater_is_better if config.greater_is_better else False
    )

# print(training_args)

class WeightDecayedTrainer(Trainer):
    def get_decay_parameter_names(self, model) -> List[str]:
        decay_parameters = get_parameter_names(model, no_decay_layer_types)
        decay_parameters = [name for name in decay_parameters if not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_names)]
        return decay_parameters

MyTrainer = WeightDecayedTrainer if config.weight_decay else Trainer

class TotalFlopsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['eval_loss']
        eval_ppl = math.exp(eval_loss)
        wandb.log({'eval_ppl': eval_ppl, 'total_flos': state.total_flos}) # , step=state.global_step)

MyCallbacks = []
if config.compute_flops: MyCallbacks += [TotalFlopsCallback()]
if config.early_stopping_patience: MyCallbacks += [EarlyStoppingCallback(config.early_stopping_patience, config.early_stopping_threshold)]

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=MyCallbacks,
    )

if config.resume_from_checkpoint:
    print('Resuming from checkpoint')
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
else:
    print('Starting training from scratch')
    trainer.train()
