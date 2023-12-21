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
import re

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import argparse
import time
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
parser.add_argument("-t", "--token_path", type=str, default="/data/chatglm2-6b-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/chatglm2-6b-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")


class ChatGLM2Demo(ChatDemo):
    # Replace English punctuation with Chinese punctuation in the Chinese sentences.
    def process_response(self, response):
        response = response.strip()
        puncts = {",": "，", "!": "！", ":": "：", ";": "；", "\?": "？"}
        for eng, chn in puncts.items():
            response = re.sub(r"([\u4e00-\u9fff])%s" % eng, r"\1%s" % chn, response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % eng, r"%s\1" % chn, response)
        return response

    # Refer to https://github.com/THUDM/ChatGLM-6B/blob/main/web_demo.py
    def create_chat_input_token(self, query, history):
        if history is None:
            history = []
        if not history:
            prompt = "[Round 0]\n问：{}\n答：".format(query)
        else:
            history = history[-2:] if len(history) > 2 else history
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        input_tokens = self.tokenizer([prompt], return_tensors="pt").input_ids
        return input_tokens

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">ChatGLM2</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = ChatGLM2Demo(args.token_path, args.model_path, dtype=args.dtype)

    demo.launch(False)
