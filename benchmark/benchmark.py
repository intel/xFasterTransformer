import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer, TextStreamer
import json
import pathlib

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--token_path", type=str, default="/data/chatglm-6b", help="Path to token file")
parser.add_argument("--model_path", type=str, default="/data/chatglm-6b/cpu", help="Path to model file")
parser.add_argument("--model_name", type=str, default="Model", help="Model name")
parser.add_argument("--prompt_path", type=str, default="./prompt_pool.json", help="Path to model file")
parser.add_argument("--token_in", type=str, default=32, help="Input Token Len")
parser.add_argument("--token_out", type=int, default=32, help="Output Token Len, MaxLen=IN+OUT")
parser.add_argument("--beam_width", type=int, default=1, help="Beam Search Width")
parser.add_argument("--input_prompt", type=str, default=None, help="Input Prompt")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--iteration", type=int, default=10, help=" Benchmakr Iterations")
parser.add_argument("--warmup", type=int, default=2, help="Warm up Iterations")
parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "int8"], default="fp16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to True.", type=boolean_string, default=True)

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")


import xfastertransformer

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.prompt_path, "r") as json_file:
        prompt_pool = json.load(json_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )
    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = (
        TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False)
        if model.rank == 0 and args.beam_width == 1
        else None
    )
    input_prompts = []
    if model.rank == 0:
        # input prompt
        print("======start=======")
        if args.input_prompt is not None:
            input_prompt = args.input_prompt
        elif args.token_in in prompt_pool:
            input_prompt = prompt_pool[args.token_in]
            print(input_prompt)
        else:
            raise SystemExit("[ERROR] Plese use --input_prompt if you want custom input.")
        for _ in range(args.batch_size):
            input_prompts.append(input_prompt)
        # Master
        input_ids = tokenizer(input_prompts, return_tensors="pt", padding=args.padding).input_ids
        input_token_nums = int(torch.numel(input_ids) / args.batch_size)
        print("Input token Length is", input_token_nums)
        print("Batch_size:", args.batch_size)
        max_len = input_token_nums + args.token_out
        print("Max_len:", max_len)
        print("=" * 50)
        # Perform 100 runs and store execution times in a list
        execution_times = []
        first_token_times = []
        remained_token_times = []
        # warm up
        for i in range(args.warmup):
            model.generate(input_ids, num_beams=args.beam_width, max_length=max_len, streamer=None)
        print("Start benchmark:")
        for i in range(args.iteration):
            print("iteration", i, ":")
            start_time = time.perf_counter()
            model.config(max_length=max_len, num_beams=args.beam_width)
            model.input(input_ids)
            # first token
            next_tokens = model.forward()
            first_token_time = time.perf_counter() - start_time
            first_token_times.append(first_token_time)
            if streamer is not None and args.beam_width == 1 and args.batch_size == 1:
                streamer.put(next_tokens.cpu())
            # remaining tokens
            start_time = time.perf_counter()
            while not model.is_done():
                next_tokens = model.forward()
                # print(next_tokens)
                if streamer is not None and args.beam_width == 1 and args.batch_size == 1:
                    streamer.put(next_tokens.cpu())
            if streamer is not None and args.beam_width == 1 and args.batch_size == 1:
                streamer.end()
            generated_ids = model.finalize()
            end_time = time.perf_counter()
            remained_token_time = time.perf_counter() - start_time
            remained_token_times.append(remained_token_time)

        output_token_nums = int(torch.numel(generated_ids) / args.batch_size) - input_token_nums
        # Sort the execution times in ascending order
        remained_token_times.sort()
        # Get the 90th element (index 89) from the sorted list
        latency_90 = remained_token_times[int(args.iteration * 0.9) - 1] * 1000 / (output_token_nums - 1)
        # Calculate the first token latency
        first_token_latency = sum(first_token_times) / len(first_token_times) * 1000
        Throughput = 1000 / latency_90 * args.batch_size
        print("\n")
        print("=" * 50 + args.model_name + " Final Performance" + "=" * 50)
        print(f"First token Latency:\t{first_token_latency:.2f} ms")
        print(f"Next token Latency:\t{latency_90:.2f} ms")
        print(f"Final Throughput:\t{Throughput:.2f} tokens/s")
    else:
        for i in range(args.warmup + args.iteration):
            model.generate()
