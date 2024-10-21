#!/usr/bin/env python

import os
import argparse
import pickle
import glob

from modeling_hyena import StripedHyenaModelForExtractingEmbeddings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
parser.add_argument('-c', '--checkpoint', type=str, help='Path to model checkpoint directory')
parser.add_argument('-d', '--dataset', type=str, help='Path to tokenized datasets')
parser.add_argument('-t', '--dataset_type', type=str, help='Type of the dataset', default='test')
parser.add_argument('-n', '--num_shards', type=int, help='Number of shards', default=256)
parser.add_argument('-b', '--per_device_eval_batch_size', type=int, help='Per device evaluation batch size', default=8)
parser.add_argument('-e', '--eval_accumulation_steps', type=int, help='Evaluation accumulation steps', default=4)
parser.add_argument('-p', '--num_threads', type=int, help='Number of threads', default=8)

args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)
vocab_size = tokenizer.vocab_size

model = StripedHyenaModelForExtractingEmbeddings.from_pretrained(args.checkpoint, tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

project_id, model_name, checkpoint_id = args.checkpoint.split('/')[-3:]
dataset_id = os.path.basename(args.dataset)
dataset_prefix = f"{dataset_id}.{args.dataset_type}_dataset"
predictions_dir = f"/scratch/goodarzilab/saberi/{project_id}/{model_name}/{checkpoint_id}/{dataset_prefix}"
os.makedirs(predictions_dir, exist_ok=True)

tokenized_datasets = load_from_disk(args.dataset)

dataset_id = list(tokenized_datasets.keys())[0]
dataset_columns = tokenized_datasets.column_names[dataset_id]
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
    output_dir=predictions_dir,
    run_name=model_name,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    eval_accumulation_steps=args.eval_accumulation_steps,
    seed=SEED,
    group_by_length=True,
    length_column_name='seq_len',
    auto_find_batch_size=True,
    dataloader_num_workers=args.num_threads,
    dataloader_persistent_workers=False)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator)

last_saved_shard = -1

sharded_pkls = glob.glob(f"{predictions_dir}/{dataset_prefix}.embeddings.shard_*.pkl")

if len(sharded_pkls) == 0:
    last_saved_shard = -1
else:
    last_saved_shard = sorted([pkl.split('.')[-2].removeprefix('shard_') for pkl in sharded_pkls])[-1]
    last_saved_shard = int(last_saved_shard)

for i in range(last_saved_shard + 1, args.num_shards):
    print(f"Processing shard {i} of {args.num_shards}")
    sharded_dataset = the_dataset.shard(args.num_shards, i, contiguous=True)

    sharded_dataset = sharded_dataset.map(lambda batch: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()})

    output = trainer.predict(sharded_dataset)

    embeds_filename = f"{predictions_dir}/{dataset_prefix}.embeddings.shard_{i}.pkl"
    with open(embeds_filename, 'wb') as f:
        pickle.dump(output.embeds_filename.cpu(), f)

    # Save transcripts
    transcripts_filename = f"{predictions_dir}/{dataset_prefix}.embeddings.shard_{i}.transcripts.pkl"
    with open(transcripts_filename, 'wb') as f:
        pickle.dump(sharded_dataset['transcript_id'], f)
