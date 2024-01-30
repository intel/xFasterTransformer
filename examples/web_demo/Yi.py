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

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import argparse
import time
import torch
from demo_utils import ChatDemo


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

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/llama-2-7b-chat-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")


class YiDemo(ChatDemo):
    # Refer to https://github.com/facebookresearch/llama/blob/main/llama/generation.py
    def create_chat_input_token(self, query, history):
        if history is None:
            history = []

        history.append((query, None))
        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            print(f"user_msg={user_msg}, model_msg={model_msg}")
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to("cpu")
        return model_inputs

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">Yi</h1>""")

    def config(self):
        return {
            "stop_words_ids": [[2], [6], [7], [8]]
        }

if __name__ == "__main__":
    args = parser.parse_args()
    demo = YiDemo(args.token_path, args.model_path, dtype=args.dtype)

    demo.launch(False)
