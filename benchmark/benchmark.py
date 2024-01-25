# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from typing import Tuple, List
# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer, TextStreamer
import json
import pathlib
import numpy as np

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = ["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8", "bf16_int4", "int4", "bf16_nf4", "nf4", "w8a8", "bf16_w8a8", "w8a8_int8", "w8a8_int4", "w8a8_nf4"]

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
parser.add_argument("--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to True.", type=boolean_string, default=True)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=False)

def build_inputs_chatglm(tokenizer, query: List[str], padding, history: List[Tuple[str, str]] = []):
    prompts = []
    for item in query:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, item)
        prompts.append(prompt)
    # print('### prompt={}'.format(prompt))
    inputs = tokenizer(prompts, return_tensors="pt", padding=padding).input_ids
    return inputs

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")


import xfastertransformer

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.prompt_path, "r") as json_file:
        prompt_pool = json.load(json_file)
    
    if "chatglm" in args.model_name.lower() :
        model_prompt=prompt_pool["chatglm"]
    if "chatglm2" in args.model_name.lower() :
        model_prompt=prompt_pool["chatglm2"]
    if "chatglm3" in args.model_name.lower() :
        model_prompt=prompt_pool["chatglm3"]
    if "llama" in args.model_name.lower() :
        model_prompt=prompt_pool["llama"]
    if "baichuan" in args.model_name.lower() :
        model_prompt=prompt_pool["baichuan"]
    if "baichuan2" in args.model_name.lower() :
        model_prompt=prompt_pool["baichuan2"]
    if "opt" in args.model_name.lower() :
        model_prompt=prompt_pool["opt"]
    if "qwen" in args.model_name.lower() :
        model_prompt=prompt_pool["qwen"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )
    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    input_prompts = []
    if model.rank == 0:
        # input prompt
        print("======start=======")
        if args.input_prompt is not None:
            input_prompt = args.input_prompt
        elif args.token_in in model_prompt:
            input_prompt = model_prompt[args.token_in]
            print(input_prompt)
        else:
            raise SystemExit("[ERROR] Plese use --input_prompt if you want custom input.")
        for _ in range(args.batch_size):
            input_prompts.append(input_prompt)
        # Master
        if args.chat and "chatglm" in args.model_name.lower() :
            if args.batch_size > 1:
                print("[INFO] chat mode only support batchsize=1")
            input_ids = build_inputs_chatglm(tokenizer, input_prompts, args.padding)
        else :
            input_ids = tokenizer(input_prompts, return_tensors="pt", padding=args.padding).input_ids
        input_token_nums = int(torch.numel(input_ids) / args.batch_size)
        if args.token_in is not None and int(args.token_in) != input_token_nums :
            print(f"[WARN] input_token_size ({input_token_nums}) != required_input_size ({args.token_in})")
        print("Input token Length is", input_token_nums)
        print("Batch_size:", args.batch_size)
        max_len = input_token_nums + args.token_out
        print("Max_len:", max_len)
        print("=" * 50)
        # Perform 100 runs and store execution times in a list
        execution_times = []
        first_token_times = []
        remained_token_times = []
        total_times = []
        # warm up
        for i in range(args.warmup):
            model.generate(input_ids, num_beams=args.beam_width, max_length=max_len, streamer=None)
        print("Start benchmark:")
        for i in range(args.iteration):
            total_time = 0.0
            print("iteration", i, ":")
            model.config(max_length=max_len, num_beams=args.beam_width)
            model.input(input_ids)
            # first token
            start_time = time.perf_counter()
            next_tokens = model.forward()
            end_time = time.perf_counter()
            first_token_time = end_time - start_time
            total_time += first_token_time
            first_token_times.append(first_token_time)
            # remaining tokens
            cost_list = []
            while not model.is_done():
                start_time = time.perf_counter()
                next_tokens = model.forward()
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                cost_list.append(end_time - start_time)
                # print(next_tokens)
            generated_ids = model.finalize()
            total_times.append(total_time)
            remained_token_times.append(sum(cost_list))

        output_token_nums = int(torch.numel(generated_ids) / args.batch_size) - input_token_nums
        # Sort the execution times in ascending order
        remained_token_times.sort()
        # Get the max, min, avg and 90th element (index 89) from the sorted list
        next_token_latency_max = remained_token_times[-1] * 1000 / (output_token_nums - 1)
        next_token_latency_min = remained_token_times[0] * 1000 / (output_token_nums - 1)
        next_token_latency_avg = np.mean(remained_token_times) * 1000 / (output_token_nums - 1)
        next_token_latency_90 = remained_token_times[int(args.iteration * 0.9) - 1] * 1000 / (output_token_nums - 1)
        # Calculate total latency
        inference_latency = sum(total_times) / len(total_times)
        # Calculate the first token latency
        first_token_latency = sum(first_token_times) / len(first_token_times) * 1000
        Next_token_throughput = 1000 / next_token_latency_90 * args.batch_size
        print("\n")
        print("=" * 50 + args.model_name + " Final Performance" + "=" * 50)
        print(f"Inference Latency:\t{inference_latency:.2f} s")
        print(f"First token Avg Latency:\t{first_token_latency:.2f} ms")
        print(f"Next token Max Latency:\t{next_token_latency_max:.2f} ms")
        print(f"Next token Min Latency:\t{next_token_latency_min:.2f} ms")
        print(f"Next token P90 Latency:\t{next_token_latency_90:.2f} ms")
        print(f"Next token Avg Latency:\t{next_token_latency_avg:.2f} ms")
        print(f"Next token Latency:\t{next_token_latency_90:.2f} ms")
        print(f"Throughput without 1st token:\t{Next_token_throughput:.2f} tokens/s")
    else:
        for i in range(args.warmup + args.iteration):
            model.generate()
