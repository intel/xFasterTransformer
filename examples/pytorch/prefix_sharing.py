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

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = ["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"]

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=DEFAULT_PROMPT, help="Input prompt")
parser.add_argument("--token_path", type=str, default="/data/llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("--model_path", type=str, default="/data/llama-2-7b-chat-cpu", help="Path to model file")
parser.add_argument("--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to True.", type=boolean_string, default=False)
parser.add_argument("--streaming", help="Streaming output, Default to True.", type=boolean_string, default=True)
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--prefix_len", help="", type=int, default=10)


def build_inputs_baichuan(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    inputs = tokenizer(query, return_tensors="pt", padding=padding).input_ids
    suffix = torch.tensor([[196]])
    prefix = torch.tensor([[195]])
    inputs = torch.cat((prefix, inputs, suffix), dim=1)
    return inputs


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

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )

    input_prompt = args.input
    if "baichuan" in args.model_path.lower():
        input_ids = build_inputs_baichuan(tokenizer, input_prompt, args.padding)
    else:
        input_ids = tokenizer(input_prompt, return_tensors="pt", padding=args.padding).input_ids

    if input_ids.shape[-1] <= args.prefix_len:
        print(
            f"[ERROR] input length should > prefix length, but input is {input_ids.shape[-1]} and prefix is {args.prefix_len}."
        )
        sys.exit(1)

    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = None
    if model.rank == 0 and args.streaming and args.num_beams == 1:
        streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False)

    # Master
    if model.rank == 0:
        print(f"[INFO] Input prompt length is  :{input_ids.shape[-1]}.")
        print(f"[INFO] Shared prefix length is :{args.prefix_len}.")
        # Base
        start_time = time.perf_counter()
        model.config(max_length=input_ids.shape[-1] + args.output_len, num_beams=args.num_beams)
        model.input(input_ids)
        # first token
        first_start_time = time.perf_counter()
        next_tokens = model.forward()
        first_end_time = time.perf_counter()
        first_token_time = first_end_time - first_start_time
        # remaining tokens
        while not model.is_done():
            next_tokens = model.forward()
        generated_ids = model.finalize()
        end_time = time.perf_counter()

        ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for snt in ret:
            print(snt)
        print("=" * 20 + "Performance" + "=" * 20)
        execution_time = end_time - start_time
        print(f"[INFO] Origin 1st token time:\t{first_token_time:.2f} s")
        print(f"[INFO] Origin execution time:\t{execution_time:.2f} s")

        # Enable perfix sharing
        truncate_tail = input_ids.shape[-1] - args.prefix_len
        model.prefix_sharing(input_ids, truncate_tail=truncate_tail)

        start_time = time.perf_counter()
        model.config(max_length=input_ids.shape[-1] + args.output_len, num_beams=args.num_beams)
        model.input(input_ids)
        # first token
        first_start_time = time.perf_counter()
        next_tokens = model.forward()
        first_end_time = time.perf_counter()
        first_token_time = first_end_time - first_start_time
        # remaining tokens
        while not model.is_done():
            next_tokens = model.forward()
        generated_ids = model.finalize()
        end_time = time.perf_counter()

        ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for snt in ret:
            print(snt)
        print("=" * 20 + "Performance" + "=" * 20)
        execution_time = end_time - start_time
        print(f"Prefix sharing 1st token time:\t{first_token_time:.2f} s")
        print(f"Prefix sharing execution time:\t{execution_time:.2f} s")

    else:
        # Slave
        model.generate()

        model.prefix_sharing()
        model.generate()
