# Copyright (c) 2023-2024 Intel Corporation
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

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer
import json
import numpy as np

import argparse
import configparser

import csv


def check_and_update_csv(file_path: str, data: dict):
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a" if file_exists else "w", newline="", encoding='utf-8-sig') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "bf16_fp16",
    "bf16_int8",
    "bf16_int4",
    "int4",
    "bf16_nf4",
    "nf4",
    "w8a8",
    "bf16_w8a8",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

KVCACHE_DTYPE_LIST = ["fp16", "int8"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, help="Model name")
parser.add_argument("--token_in", type=str, default="32", help="Input Token Len")
parser.add_argument("--token_out", type=int, default=32, help="Output Token Len, MaxLen=IN+OUT")
parser.add_argument("--beam_width", type=int, default=1, help="Beam Search Width")
parser.add_argument("--input_prompt", type=str, default=None, help="Input Prompt")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--iteration", type=int, default=3, help=" Benchmakr Iterations")
parser.add_argument("--warmup", type=int, default=1, help="Warm up Iterations")
parser.add_argument("--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")
parser.add_argument("--token_path", type=str, default=None, help="Path to token file")
parser.add_argument("--model_path", type=str, default=None, help="Path to model file")
parser.add_argument("--prompt_path", type=str, default="prompt.json", help="Path to model file")
parser.add_argument("--padding", help="Enable padding, Default to True.", type=boolean_string, default=True)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=False)
parser.add_argument("--csv", type=str, default="", help="Path to csv file")


def build_inputs_chatglm(tokenizer, query: List[str], padding, history: List[Tuple[str, str]] = []):
    prompts = []
    for item in query:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, item)
        prompts.append(prompt)
    inputs = tokenizer(prompts, return_tensors="pt", padding=padding).input_ids
    return inputs


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.prompt_path, "r") as json_file:
        prompt_pool = json.load(json_file)

    if not args.model_name:
        args.model_name = os.path.basename(os.path.dirname(args.model_path))
    else:
        os.environ.setdefault("XFT_FAKE_MODEL", "1")

    # if enabled the XFT_FAKE_MODEL = 1, will try to load real weight from config.ini:model_name
    if os.environ.get("XFT_FAKE_MODEL", "-1") == "1":
        model_path_suffix = "-xft"
        _config = configparser.ConfigParser()
        _config.read(os.path.join(args.model_path, "config.ini"))
        _weight_path = _config[_config.sections()[0]]["model_name"].rstrip(os.path.sep) + model_path_suffix
        if os.path.exists(_weight_path):
            print(f"The folder '{_weight_path}' exists. Loading real weight!")
            os.environ.update({"XFT_FAKE_MODEL": "0"})
            args.model_path = _weight_path
        else:
            print(f"The folder '{_weight_path}' does not exist. Using fake weight!")

    if "chatglm" in args.model_name.lower():
        model_prompt = prompt_pool["chatglm"]
    if "chatglm2" in args.model_name.lower():
        model_prompt = prompt_pool["chatglm2"]
    if "chatglm3" in args.model_name.lower():
        model_prompt = prompt_pool["chatglm3"]
    if "llama" in args.model_name.lower():
        model_prompt = prompt_pool["llama"]
    if "deepseek" in args.model_name.lower():
        model_prompt = prompt_pool["llama"]
    if "gemma" in args.model_name.lower():
        model_prompt = prompt_pool["llama"]
    if "baichuan" in args.model_name.lower():
        model_prompt = prompt_pool["baichuan"]
    if "baichuan2" in args.model_name.lower():
        model_prompt = prompt_pool["baichuan2"]
    if "opt" in args.model_name.lower():
        model_prompt = prompt_pool["opt"]
    if "qwen" in args.model_name.lower():
        model_prompt = prompt_pool["qwen"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True, legacy=False
    )

    try:
        import xfastertransformer

        print("[INFO] xfastertransformer is installed, using pip installed package.")
    except Exception as e:
        sys.path.append("../src")
        import xfastertransformer

        print("[INFO] xfastertransformer is not installed in pip, using source code.")

    model = xfastertransformer.AutoModel.from_pretrained(
        args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
    )
    input_prompts = []
    if model.rank == 0:
        # input prompt
        print("======start=======")
        print("[INFO] input argparse = ", args)
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
        if args.chat and "chatglm" in args.model_name.lower():
            if args.batch_size > 1:
                print("[INFO] chat mode only support batchsize=1")
            input_ids = build_inputs_chatglm(tokenizer, input_prompts, args.padding)
        else:
            input_ids = tokenizer(input_prompts, return_tensors="pt", padding=args.padding).input_ids
        input_token_nums = int(torch.numel(input_ids) / args.batch_size)
        if args.token_in is not None and int(args.token_in) != input_token_nums:
            print(f"[WARN] input_token_size ({input_token_nums}) != required_input_size ({args.token_in})")
        print("Input token Length is", input_token_nums)
        print("Batch_size:", args.batch_size)
        max_len = input_token_nums + args.token_out
        print("Max_len:", max_len)
        print("=" * 50)
        # Perform 100 runs and store execution times in a list
        execution_times = []
        first_token_times = []
        next_token_times = []
        total_times = []
        # warm up
        for i in range(args.warmup):
            model.generate(input_ids, num_beams=args.beam_width, max_length=max_len, streamer=None)

        print("Start benchmark:")
        for i in range(args.iteration):
            print("iteration", i, ":")
            model.config(max_length=max_len, num_beams=args.beam_width)
            model.input(input_ids)
            # first token
            start_time = time.perf_counter()
            next_tokens = model.forward()
            first_token_time = time.perf_counter() - start_time
            first_token_times.append(first_token_time)
            # remaining tokens
            cost_list = []
            token_list = [next_tokens.view(-1).tolist()[0]]
            while not model.is_done():
                start_time = time.perf_counter()
                next_tokens = model.forward()
                next_time = time.perf_counter() - start_time
                cost_list.append(next_time)
                token_list.append(next_tokens.view(-1).tolist()[0])
            generated_ids = model.finalize()
            total_times.append(first_token_time + sum(cost_list))
            next_token_times += cost_list
            response = tokenizer.decode(token_list, skip_special_tokens=True)
            print(f"    Response: {response}")

        total_times = list(map(lambda x: x * 1000, total_times))
        first_token_times = list(map(lambda x: x * 1000, first_token_times))
        next_token_times = list(map(lambda x: x * 1000, next_token_times))

        print("\n")
        print("=" * 50 + args.model_name + " Final Performance" + "=" * 50)
        print(f"Inference Avg Latency:\t{np.mean(total_times):.2f} ms")
        print(f"First token Avg Latency:\t{np.mean(first_token_times):.2f} ms")
        print(f"Next token Max Latency:\t{np.max(next_token_times):.2f} ms")
        print(f"Next token Min Latency:\t{np.min(next_token_times):.2f} ms")
        print(f"Next token P90 Latency:\t{np.percentile(next_token_times, 90):.2f} ms")
        print(f"Next token Avg Latency:\t{np.mean(next_token_times):.2f} ms")
        print(f"Next token Latency:\t{np.percentile(next_token_times, 90):.2f} ms")
        print(f"Throughput without 1st token:\t{1000 / np.mean(next_token_times) * args.batch_size:.2f} tokens/s")
        print("=" * 120, "\n" * 3)

        if args.csv != "":
            from datetime import datetime

            arg_dict = dict(
                filter(
                    lambda item: str(item[0]).find("path") == -1 and str(item[0]).find("csv") == -1, vars(args).items()
                )
            )

            rst = {
                "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "infer_avg(ms)": round(np.mean(total_times), 2),
                "1st_avg(ms)": round(np.mean(first_token_times), 2),
                "2nd_max(ms)": round(np.max(next_token_times), 2),
                "2nd_min(ms)": round(np.min(next_token_times), 2),
                "2nd_P90(ms)": round(np.percentile(next_token_times, 90), 2),
                "2nd_avg(ms)": round(np.mean(next_token_times), 2),
                "throughput_wo_1st (tokens/s)": round(1000 / np.mean(next_token_times) * args.batch_size, 2),
                **arg_dict,
                "Fake_model": True if os.environ.get("XFT_FAKE_MODEL", "-1") == "1" else False,
                "Response": response,
            }
            # print(rst)
            check_and_update_csv(args.csv, rst)

    else:
        for i in range(args.warmup + args.iteration):
            model.generate()
