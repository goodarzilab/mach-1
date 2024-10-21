import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from modeling_hyena import StripedHyenaModelForCausalLM

from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast,
    pipeline
)

random.seed(42)
np.random.seed(42)
set_seed(42)
transformers_set_seed(42)

checkpoint = '/home/saberi/projects/lornash/checkpoint-final'
model = StripedHyenaModelForCausalLM.from_pretrained(checkpoint)
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

generator = pipeline(task='text-generation', model=model, tokenizer=tokenizer, device=0)

generations_dir = "generations"
os.makedirs(generations_dir, exist_ok=True)

batch_size = 100
index = 1

if __name__ == "__main__":
    try:
        while True:
            prompt = "HS"
            generated_sequences = generator(
                prompt,
                num_return_sequences=batch_size,
                max_new_tokens=10_000,
                do_sample=True,
                stop_strings=["E"],
                eos_token_id=tokenizer.convert_tokens_to_ids("E")
            )
            output_file = os.path.join(generations_dir, f"generated_batch_{index}.txt")
            with open(output_file, "w") as f:
                for sequence in generated_sequences:
                    f.write(sequence['generated_text'] + "\n")
            index += 1

    except KeyboardInterrupt:
        print("Process interrupted.")
