#!/usr/bin/env python

import os
import argparse
import pickle
import glob

from configuration_hyena import StripedHyenaConfig
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
    Trainer,
)

SEED = 42
set_seed(SEED)
transformers_set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--checkpoint", type=str, help="Path to model checkpoint directory"
)
parser.add_argument("--tokenizer", type=str, help="Path to tokenizer json file")
parser.add_argument(
    "--max_seq_len", type=int, help="Maximum sequence length", default=2 ** 16
)
parser.add_argument("-d", "--dataset", type=str, help="Path to tokenized datasets")
parser.add_argument(
    "-t", "--dataset_type", type=str, help="Type of the dataset", default="test"
)
parser.add_argument("--predictions_dir", type=str, default="predictions")
parser.add_argument(
    "-n", "--num_shards", type=int, help="Number of shards", default=256
)
parser.add_argument(
    "-b",
    "--per_device_eval_batch_size",
    type=int,
    help="Per device evaluation batch size",
    default=8,
)
parser.add_argument(
    "-e",
    "--eval_accumulation_steps",
    type=int,
    help="Evaluation accumulation steps",
    default=4,
)
parser.add_argument(
    "-p", "--num_threads", type=int, help="Number of threads", default=8
)

args = parser.parse_args()

config_dict = {
    "vocab_size": 32,  # Effective vocabulary size (from RNA tokens in the paper)
    "hidden_size": 128,  # Model width per block (from the architecture details)
    "num_filters": 128,  # Filters align with model width
    "inner_mlp_size": 352,  # Approximation; might require refinement from tensor inspection
    "attn_layer_idxs": [
        4,
        8,
        12,
    ],  # Interleaved attention layers (as described in the architecture)
    "hyena_layer_idxs": [
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
    ],  # Remaining layers are Hyena layers
    "num_layers": 16,  # Total number of layers
    "tie_embeddings": True,  # No mention of tied embeddings
    "short_filter_length": 3,  # Typical short filter length
    "num_attention_heads": 16,  # Guess for attention heads; adjust as necessary
    "proj_groups": 1,  # Default grouping for projections
    "hyena_filter_groups": 1,  # Based on typical Hyena usage
    "split_k0": True,  # Split kernel optimization
    "column_split_hyena": True,  # Optimize for column splits
    "column_split": False,  # Default unless specified
    "model_parallel_size": 1,  # Single GPU (adjust based on hardware)
    "pipe_parallel_size": 1,  # Single pipeline (adjust if multi-pipeline is known)
    "short_filter_bias": True,  # Default
    "mha_out_proj_bias": True,  # Default for Hyena
    "qkv_proj_bias": True,  # Default
    "final_norm": True,  # Normalize at the end
    "use_cache": False,  # Default
    "use_flash_attention_2": True,  # Optimized attention mechanism
    "use_flash_rmsnorm": True,  # Efficient normalization
    "use_flash_depthwise": False,  # Default unless explicitly mentioned
    "use_flashfft": False,  # Hyena leverages FFT for speed
    "inference_mode": True,  # Default for training
    "prefill_style": "fft",  # FFT-based prefill (specific to Hyena-style computation)
    "max_seqlen": 65536,  # Maximum context length (65k as stated in the paper)
    "eps": 1e-5,  # Default epsilon for numerical stability
    "state_size": 8,  # Default
    "rotary_emb_base": 500000,  # Rotary embeddings
    "smeared_gqa": False,  # Default unless specified
    "make_vocab_size_divisible_by": 8,  # Common optimization for efficient computation
    "log_intermediate_values": False,  # Default
    "bidirectional": False,  # Causal model
}

config = StripedHyenaConfig(**config_dict)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=args.tokenizer,
    padding_side="right",
    truncation_side="right",
    cls_token="[CLS]",
    bos_token="[CLS]",
    sep_token="[SEP]",
    eos_token="[SEP]",
    unk_token="[UNK]",
    mask_token="[MASK]",
    pad_token="[PAD]",
    model_max_length=args.max_seq_len,
)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)
vocab_size = tokenizer.vocab_size

model = StripedHyenaModelForExtractingEmbeddings.from_pretrained(
    args.checkpoint, tokenizer=tokenizer, config=config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# project_id, model_name, checkpoint_id = args.checkpoint.split('/')[-3:]
project_id, model_name, checkpoint_id = (
    "lornash",
    "StripedHyenaModelForCausalLM",
    "model.safetensors",
)
dataset_id = os.path.basename(args.dataset)
dataset_prefix = f"{dataset_id}.{args.dataset_type}_dataset"

predictions_dir = args.predictions_dir
os.makedirs(predictions_dir, exist_ok=True)

tokenized_datasets = load_from_disk(args.dataset)

dataset_id = list(tokenized_datasets.keys())[0]
dataset_columns = tokenized_datasets.column_names[dataset_id]
columns_to_keep = [
    "input_ids",
    "attention_mask",
    "special_tokens_mask",
    "seq_len",
    "transcript_id",
]
columns_to_keep = list(set(dataset_columns) & set(columns_to_keep))
tokenized_datasets = tokenized_datasets.select_columns(columns_to_keep)

if args.dataset_type not in ["train", "validation", "test"]:
    tokenized_datasets[args.dataset_type] = concatenate_datasets(
        [v for _, v in tokenized_datasets.items()]
    )

the_dataset = tokenized_datasets[args.dataset_type]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=predictions_dir,
    run_name=model_name,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    eval_accumulation_steps=args.eval_accumulation_steps,
    seed=SEED,
    group_by_length=True,
    length_column_name="seq_len",
    auto_find_batch_size=True,
    dataloader_num_workers=args.num_threads,
    dataloader_persistent_workers=False,
    report_to=[],
)

trainer = Trainer(
    model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator
)

last_saved_shard = -1

sharded_pkls = glob.glob(f"{predictions_dir}/{dataset_prefix}.embeddings.shard_*.pkl")

if len(sharded_pkls) == 0:
    last_saved_shard = -1
else:
    last_saved_shard = sorted(
        [pkl.split(".")[-2].replace("shard_", "") for pkl in sharded_pkls]
    )[-1]
    last_saved_shard = int(last_saved_shard)

for i in range(last_saved_shard + 1, args.num_shards):
    print(f"Processing shard {i} of {args.num_shards}")
    sharded_dataset = the_dataset.shard(args.num_shards, i, contiguous=True)

    sharded_dataset = sharded_dataset.map(
        lambda batch: {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
    )

    output = trainer.predict(sharded_dataset)

    embeds_filename = f"{predictions_dir}/{dataset_prefix}.embeddings.shard_{i}.pkl"
    with open(embeds_filename, "wb") as f:
        pickle.dump(output.predictions, f)

    # Save transcripts
    transcripts_filename = (
        f"{predictions_dir}/{dataset_prefix}.embeddings.shard_{i}.transcripts.pkl"
    )
    with open(transcripts_filename, "wb") as f:
        pickle.dump(sharded_dataset["transcript_id"], f)
