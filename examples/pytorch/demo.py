import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer, TextStreamer

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--token_path", type=str, default="/data/opt-13b", help="Path to token file")
parser.add_argument("--model_path", type=str, default="/data/1-gpu", help="Path to model file")
parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "int8"], default="fp16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to True.", type=boolean_string, default=False)

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )

    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False) if model.rank == 0 else None

    if model.rank == 0:
        # Master
        while True:
            input_prompt = input("\nPlease enter the prompt: ")
            if input_prompt == "":
                input_prompt = DEFAULT_PROMPT
                print("[Use default prompt]:" + input_prompt)
            input_ids = tokenizer(input_prompt, return_tensors="pt", padding=args.padding).input_ids
            print("=" * 50)

            start_time = time.perf_counter()
            generated_ids = model.generate(input_ids, max_length=193, streamer=streamer)
            end_time = time.perf_counter()

            if streamer is None:
                ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for snt in ret:
                    print(snt)
            print("=" * 20 + "Performance" + "=" * 20)
            execution_time = end_time - start_time
            print(f"Execution time:\t{execution_time:.2f} s")
            input_token_nums = torch.numel(input_ids)
            output_token_nums = torch.numel(generated_ids) - input_token_nums
            latency = execution_time * 1000 / output_token_nums
            througput = output_token_nums / execution_time
            print(f"Latency:\t{latency:.2f} ms/token")
            print(f"Througput:\t{througput:.2f} tokens/s")
    else:
        # Slave
        while True:
            model.generate()
