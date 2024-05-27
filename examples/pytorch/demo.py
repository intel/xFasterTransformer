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
import sys
from typing import Tuple, List

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
import json
import traceback
import transformers
from transformers import AutoTokenizer, TextStreamer
from transformers import PreTrainedTokenizer

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

KVCACHE_DTYPE_LIST = ["fp16", "int8"]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/chatglm-6b-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/chatglm-6b-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")
parser.add_argument("--padding", help="Enable padding, Default to False.", type=boolean_string, default=False)
parser.add_argument("--streaming", help="Streaming output, Default to True.", type=boolean_string, default=True)
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("-o", "--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=False)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=False)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)
parser.add_argument("--rep_penalty", help="param for repetition penalty. 1.0 means no penalty", type=float, default=1.0)


def check_transformers_version_compatibility(token_path):
    config_path = os.path.join(token_path, "config.json")
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)

        transformers_version = config_data.get("transformers_version")
    except Exception as e:
        pass
    else:
        if transformers_version:
            if transformers.__version__ != transformers_version:
                print(
                    f"[Warning] The version of `transformers` in model configuration is {transformers_version}, and version installed is {transformers.__version__}. "
                    + "This tokenizer loading error may be caused by transformers version compatibility. "
                    + f"You can downgrade or reinstall transformers by `pip install transformers=={transformers_version} --force-reinstall` and try again."
                )


def build_inputs_chatglm(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer(prompt, return_tensors="pt", padding=padding).input_ids
    return inputs


def build_inputs_baichuan(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    inputs = tokenizer(query, return_tensors="pt", padding=padding).input_ids
    suffix = torch.tensor([[196]])
    prefix = torch.tensor([[195]])
    inputs = torch.cat((prefix, inputs, suffix), dim=1)
    return inputs


def build_inputs_llama(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    inputs = tokenizer([f"[INST] {query.strip()} [/INST]"], return_tensors="pt", padding=padding).input_ids
    return inputs


def build_inputs_qwen(
    tokenizer: PreTrainedTokenizer,
    query: str,
    padding,
    history: List[Tuple[str, str]] = None,
    system: str = "You are a helpful assistant.",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set()) + nl_tokens + tokenizer.encode(
                content, allowed_special=set()
            )

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"

            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return torch.tensor([context_tokens])


def build_inputs_qwen2(
    tokenizer: PreTrainedTokenizer,
    query: str,
    padding,
    history: List[Tuple[str, str]] = None,
    system: str = "You are a helpful assistant.",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    _history = history + [(query, None)]
    messages = []
    for idx, (user_msg, model_msg) in enumerate(_history):
        print(f"user_msg={user_msg}, model_msg={model_msg}")
        if idx == len(_history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "model", "content": model_msg})

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    return model_inputs


def get_stop_words_ids_qwen(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
        )
    except Exception as e:
        traceback.print_exc()
        print("[ERROR] An exception occurred during the tokenizer loading process.")
        # print(f"{type(e).__name__}: {str(e)}")
        check_transformers_version_compatibility(args.token_path)
        sys.exit(-1)

    model = xfastertransformer.AutoModel.from_pretrained(
        args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
    )
    streamer = None
    stop_words_ids = None
    if model.rank == 0 and args.streaming and args.num_beams == 1:
        streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=args.chat)

    if model.rank == 0:
        # Master
        while True:
            input_prompt = input("\nPlease enter the prompt: ")
            if input_prompt == "":
                input_prompt = DEFAULT_PROMPT
                print("[Use default prompt]:" + input_prompt)

            if args.chat and "chatglm" in args.model_path.lower():
                input_ids = build_inputs_chatglm(tokenizer, input_prompt, args.padding)
            elif "baichuan" in args.model_path.lower():
                input_ids = build_inputs_baichuan(tokenizer, input_prompt, args.padding)
            elif "qwen1.5" in args.model_path.lower() or "qwen2" in args.model_path.lower():
                input_ids = build_inputs_qwen2(tokenizer, input_prompt, args.padding)
                # https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/blob/main/generation_config.json#L6-L7
                stop_words_ids = [[151643], [151645]]
            elif "qwen" in args.model_path.lower() and ("chat" in args.model_path.lower() or args.chat):
                input_ids = build_inputs_qwen(tokenizer, input_prompt, args.padding)
                stop_words_ids = get_stop_words_ids_qwen("chatml", tokenizer)
            elif args.chat and "llama" in args.model_path.lower():
                input_ids = build_inputs_llama(tokenizer, input_prompt, args.padding)
            else:
                input_ids = tokenizer(input_prompt, return_tensors="pt", padding=args.padding).input_ids
            print("=" * 50)

            start_time = time.perf_counter()
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + args.output_len,
                streamer=streamer,
                num_beams=args.num_beams,
                stop_words_ids=stop_words_ids,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            )
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
