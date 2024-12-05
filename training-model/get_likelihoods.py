#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from safetensors.torch import load_file

from configuration_hyena import StripedHyenaConfig
from modeling_hyena import StripedHyenaModelForCausalLM

from datasets import load_from_disk, concatenate_datasets
from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

SEED = 42
set_seed(SEED)
transformers_set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint directory')
parser.add_argument('--tokenizer', type=str, help='Path to tokenizer json file')
parser.add_argument('--max_seq_len', type=int, help='Maximum sequence length', default=2**16)
parser.add_argument('--dataset', type=str, help='Path to tokenized datasets')
parser.add_argument('--dataset_type', type=str, help='Type of the dataset')
parser.add_argument('--predictions_dir', type=str, default="predictions")
parser.add_argument('--num_shards', type=int, help='Number of shards', default=32)
parser.add_argument('--per_device_eval_batch_size', type=int, help='Per device evaluation batch size', default=32)
parser.add_argument('--eval_accumulation_steps', type=int, help='Evaluation accumulation steps', default=128)
parser.add_argument('--num_threads', type=int, help='Number of threads', default=32)

args = parser.parse_args()

config_dict = {
    "vocab_size": 32,                   # Effective vocabulary size (from RNA tokens in the paper)
    "hidden_size": 128,                 # Model width per block (from the architecture details)
    "num_filters": 128,                 # Filters align with model width
    "inner_mlp_size": 352,             # Approximation; might require refinement from tensor inspection
    "attn_layer_idxs": [4, 8, 12],      # Interleaved attention layers (as described in the architecture)
    "hyena_layer_idxs": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15], # Remaining layers are Hyena layers
    "num_layers": 16,                   # Total number of layers
    "tie_embeddings": True,            # No mention of tied embeddings
    "short_filter_length": 3,           # Typical short filter length
    "num_attention_heads": 16,           # Guess for attention heads; adjust as necessary
    "proj_groups": 1,                   # Default grouping for projections
    "hyena_filter_groups": 1,           # Based on typical Hyena usage
    "split_k0": True,                   # Split kernel optimization
    "column_split_hyena": True,         # Optimize for column splits
    "column_split": False,              # Default unless specified
    "model_parallel_size": 1,           # Single GPU (adjust based on hardware)
    "pipe_parallel_size": 1,            # Single pipeline (adjust if multi-pipeline is known)
    "short_filter_bias": True,          # Default
    "mha_out_proj_bias": True,         # Default for Hyena
    "qkv_proj_bias": True,             # Default
    "final_norm": True,                 # Normalize at the end
    "use_cache": False,                  # Default
    "use_flash_attention_2": True,      # Optimized attention mechanism
    "use_flash_rmsnorm": True,          # Efficient normalization
    "use_flash_depthwise": False,       # Default unless explicitly mentioned
    "use_flashfft": False,               # Hyena leverages FFT for speed
    "inference_mode": True,            # Default for training
    "prefill_style": "fft",             # FFT-based prefill (specific to Hyena-style computation)
    "max_seqlen": 65536,                # Maximum context length (65k as stated in the paper)
    "eps": 1e-5,                        # Default epsilon for numerical stability
    "state_size": 8,                    # Default
    "rotary_emb_base": 500000,          # Rotary embeddings
    "smeared_gqa": False,               # Default unless specified
    "make_vocab_size_divisible_by": 8,  # Common optimization for efficient computation
    "log_intermediate_values": False,   # Default
    "bidirectional": False              # Causal model
}

config = StripedHyenaConfig(**config_dict)
model = StripedHyenaModelForCausalLM(config)
state_dict = load_file(args.checkpoint)

print("model state_dict:", [k for k in model.state_dict().keys() if "blocks" not in k])
print("model.safetensors:", [k for k in state_dict.keys() if "blocks" not in k])
model.load_state_dict(state_dict, strict=False) # missing backbone.unembed.weight
# model = StripedHyenaModelForCausalLM.from_pretrained(args.checkpoint)
# project_id, model_name, checkpoint_id = args.checkpoint.split('/')[-3:]
project_id, model_name, checkpoint_id = "lornash", "StripedHyenaModelForCausalLM", "model.safetensors"
dataset_id = os.path.basename(args.dataset)
dataset_prefix = f"{dataset_id}.{args.dataset_type}_dataset"

predictions_dir = args.predictions_dir
os.makedirs(predictions_dir, exist_ok=True)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=args.tokenizer,
    padding_side='right',
    truncation_side='right',
    cls_token='[CLS]',
    bos_token='[CLS]',
    sep_token='[SEP]',
    eos_token='[SEP]',
    unk_token='[UNK]',
    mask_token='[MASK]',
    pad_token='[PAD]',
    model_max_length=args.max_seq_len
)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)
vocab_size = tokenizer.vocab_size

tokenized_datasets = load_from_disk(args.dataset)

first_dataset = list(tokenized_datasets.keys())[0]
dataset_columns = tokenized_datasets.column_names[first_dataset]
columns_to_keep = ['input_ids', 'attention_mask', 'special_tokens_mask', 'seq_len', 'transcript_id']
columns_to_keep = list(set(dataset_columns) & set(columns_to_keep))
tokenized_datasets = tokenized_datasets.select_columns(columns_to_keep)

if args.dataset_type not in ['train', 'validation', 'test']:
    tokenized_datasets[args.dataset_type] = concatenate_datasets([v for _, v in tokenized_datasets.items()])

the_dataset = tokenized_datasets[args.dataset_type]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)

training_args = TrainingArguments(
    output_dir = predictions_dir,
    run_name = model_name,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    eval_accumulation_steps = args.eval_accumulation_steps,
    seed = SEED,
    group_by_length = True,
    length_column_name = 'seq_len',
    auto_find_batch_size = True,
    dataloader_num_workers = args.num_threads,
    dataloader_persistent_workers = False,
    report_to = []
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator
)

sharded_csvs = glob.glob(f"{predictions_dir}/{dataset_prefix}.likelihoods.shard_*.csv")

if len(sharded_csvs) == 0:
    last_saved_shard = -1
else:
    last_saved_shard = sorted([csv.split('.')[-2].replace('shard_','') for csv in sharded_csvs])[-1]
    last_saved_shard = int(last_saved_shard)

for i in range(last_saved_shard + 1, args.num_shards):
    print(f"Processeing shard {i} of {args.num_shards}")

    sharded_dataset = the_dataset.shard(args.num_shards, i, contiguous=True)

    output = trainer.predict(sharded_dataset)

    logits, labels = torch.from_numpy(output.predictions), torch.from_numpy(output.label_ids)

    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1] 
    labels = labels[:, 1:]

    assert(softmax_logprobs.shape[1] == labels.shape[1])

    labels[labels == -100] = tokenizer.pad_token_id

    logprobs = torch.gather(
        softmax_logprobs,
        2,
        labels.unsqueeze(-1)
    ).squeeze(-1)

    softmax_logprobs = softmax_logprobs.to(torch.float64)
    logprobs = logprobs.to(torch.float64)

    seq_likelihood = [torch.sum(logprobs[i, labels[i] != tokenizer.pad_token_id]) for i in range(len(labels))]
    seq_likelihood_float = [float(ll) for ll in seq_likelihood]
    
    tss_id, tts_id = tokenizer.convert_tokens_to_ids(['S', 'E'])

    tss_index = torch.where(labels == tss_id)
    tss_logprobs = logprobs[tss_index[0], tss_index[1]]
    tss_logprobs = tss_logprobs.tolist()

    tts_index = torch.where(labels == tts_id)
    tts_logprobs = logprobs[tts_index[0], tts_index[1]]
    tts_logprobs = tts_logprobs.tolist()

    tr_lens = np.array(tts_index[1]) - np.array(tss_index[1]) + 1

    seq_likelihood_float = [float(ll) for ll in seq_likelihood]
    tss_logprobs_float = [float(ll) for ll in tss_logprobs]
    tts_logprobs_float = [float(ll) for ll in tts_logprobs]

    seq_lls = {
        'transcript_id': sharded_dataset['transcript_id'],
        'seq_len': sharded_dataset['seq_len'],
        'tr_len': tr_lens.tolist(),
        'likelihood': seq_likelihood_float,
        'tss_logprobs': tss_logprobs_float,
        'tts_logprobs': tts_logprobs_float
        }

    seq_lls = pd.DataFrame(seq_lls)

    include_header = True if i == 0 else False
    output_suffix = str(i).zfill(len(str(args.num_shards)))
    seq_lls_filename = f"{predictions_dir}/{dataset_prefix}.likelihoods.shard_{output_suffix}.csv"
    seq_lls.to_csv(seq_lls_filename, index=False, header=include_header)

print("Done!")