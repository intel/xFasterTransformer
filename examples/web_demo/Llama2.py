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
import torch
from demo_utils import ChatDemo, XFT_DTYPE_LIST, XFT_KVCACHE_DTYPE_LIST


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/llama-2-7b-chat-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=XFT_DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=XFT_KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
EOS_ID = 2


class Llama2Demo(ChatDemo):
    # Refer to https://github.com/facebookresearch/llama/blob/main/llama/generation.py
    def create_chat_input_token(self, query, history):
        tokens = []
        if history is None:
            history = []
        if not history:
            input_tokens = self.tokenizer([f"{B_INST} {query.strip()} {E_INST}"], return_tensors="pt").input_ids
        else:
            history = history[-2:] if len(history) > 2 else history
        for i, (old_query, response) in enumerate(history):
            tokens.append(
                self.tokenizer(
                    [f"{B_INST} {old_query.strip()} {E_INST} {response.strip()} "], return_tensors="pt"
                ).input_ids
            )
            tokens.append(torch.tensor([[EOS_ID]]))
        tokens.append(self.tokenizer([f"{B_INST} {query.strip()} {E_INST}"], return_tensors="pt").input_ids)
        input_tokens = torch.cat(tokens, dim=1)
        return input_tokens

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">Llama2</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = Llama2Demo(args.token_path, args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype)

    demo.launch(False)
