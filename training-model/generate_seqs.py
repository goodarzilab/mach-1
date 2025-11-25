#!/usr/bin/env python
import os
import argparse
import torch

from scripts.modeling_mach import StripedHyenaModelForCausalLM
from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast,
    pipeline,
)

set_seed(42)
transformers_set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sequences with specified configuration')
    parser.add_argument('--prompt', type=str, default='[CLS]H', help='Generation prompt (default: [CLS]H)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=4, help='Top-k sampling (default: 4)')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling (default: 0.95)')
    parser.add_argument('--min_p', type=float, default=0.1, help='Min-p sampling (default: 0.1)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty (default: 1.0)')
    parser.add_argument('--num_sequences', type=int, default=8192, help='Total number of sequences to generate (default: 8192)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for generation (default: 512)')
    parser.add_argument('--min_new_tokens', type=int, default=1200, help='Minimum new tokens (default: 1200)')
    parser.add_argument('--max_new_tokens', type=int, default=1500, help='Maximum new tokens (default: 1500)')
    parser.add_argument('--checkpoint', type=str, 
                        default='/scratch/goodarzilab/saberi/mach/mach-1/model/mach1_gen/mach-1-v00-gen-v03/checkpoint-14336',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated from checkpoint)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    checkpoint = args.checkpoint
    model = StripedHyenaModelForCausalLM.from_pretrained(checkpoint)
    
    if args.output_dir:
        generations_dir = args.output_dir
    else:
        model_name, checkpoint_id = checkpoint.split('/')[-2:]
        generations_dir = f"/scratch/goodarzilab/saberi/mach/mach-1/generations/{model_name}/{checkpoint_id}"
    os.makedirs(generations_dir, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

    generator = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0,
        add_special_tokens=False
    )

    vocab = tokenizer.get_vocab()
    suppress_tokens = sorted([id for token, id in vocab.items() if token not in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'P']])
    begin_suppress_tokens = tokenizer.convert_tokens_to_ids(['a', 'c', 'g', 't'])
    forced_eos_token_id = tokenizer.convert_tokens_to_ids('P')

    temp = args.temperature
    tk = args.top_k
    tp = args.top_p
    mp = args.min_p
    rp = args.repetition_penalty
    
    print(f"Prompt: {args.prompt}")
    print(f"Configuration :: Temperature = {temp} :: Top_k = {tk} :: Top_p = {tp} :: Min_p = {mp} :: Repetition_penalty = {rp}")
    
    all_gen_seqs = []
    for batch_start in range(0, args.num_sequences, args.batch_size):
        current_batch_size = min(args.batch_size, args.num_sequences - batch_start)
        print(f"Generating batch {batch_start // args.batch_size + 1} ({batch_start + 1}-{batch_start + current_batch_size} of {args.num_sequences})")
        
        raw_gen_seqs = generator(
            args.prompt,
            num_return_sequences=current_batch_size,
            do_sample=True,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            min_p=mp,
            repetition_penalty=rp,
            renormalize_logits=True,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            forced_eos_token_id=forced_eos_token_id
        )
        
        batch_gen_seqs = [raw_gen_seqs[i]['generated_text'].replace('[CLS]', '').replace('[SEP]', '').replace(' ', '') for i in range(current_batch_size)]
        all_gen_seqs.extend(batch_gen_seqs)

    file_name = f"gen_T{temp}_k{tk}_p{tp}_minp{mp}_repp{rp}.txt"
    file_path = os.path.join(generations_dir, file_name)
    with open(file_path, 'w') as f:
        for seq in all_gen_seqs:
            f.write(seq + '\n')
    
    print(f"Generated {len(all_gen_seqs)} sequences saved to {file_path}")
