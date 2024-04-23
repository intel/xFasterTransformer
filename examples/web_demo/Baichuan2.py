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

USER_TOKEN_ID = 195
ASSISTANT_TOKEN_ID = 196
user_id_tensor = torch.tensor([[USER_TOKEN_ID]])
assist_id_tensor = torch.tensor([[ASSISTANT_TOKEN_ID]])


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/Baichuan2-7B-Chat-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/Baichuan2-7B-Chat-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=XFT_DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=XFT_KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")


class BaiChuan2Demo(ChatDemo):
    # Refer to https://github.com/baichuan-inc/Baichuan2/blob/main/web_demo.py
    def create_chat_input_token(self, query, history):
        input_tokens = self.tokenizer(query, return_tensors="pt").input_ids
        input_tokens = torch.cat((user_id_tensor, input_tokens, assist_id_tensor), dim=1)
        if history is None:
            history = []
        if history:
            history = history[-2:] if len(history) > 2 else history
            for i, (old_query, response) in enumerate(history):
                query_ids = self.tokenizer(old_query, return_tensors="pt").input_ids
                response_ids = self.tokenizer(response, return_tensors="pt").input_ids
                input_tokens = torch.cat(
                    (user_id_tensor, query_ids, assist_id_tensor, response_ids, input_tokens), dim=1
                )

        return input_tokens

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">Baichuan2</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = BaiChuan2Demo(args.token_path, args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype)

    demo.launch(False)
