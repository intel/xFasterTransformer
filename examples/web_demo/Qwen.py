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
import torch
from demo_utils import ChatDemo


DTYPE_LIST = ["fp16", "bf16", "int8", "int4", "bf16_fp16", "bf16_int8", "bf16_int4"]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/Qwen-7B-Chat-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/Qwen-7B-Chat-xft", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")


class QwenDemo(ChatDemo):
    # Replace English punctuation with Chinese punctuation in the Chinese sentences.
    def process_response(self, response):
        lines = response.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split("`")
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f"<br></code></pre>"
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", r"\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>" + line
        response = "".join(lines)
        response = response.rstrip("human:")
        return response

    def create_chat_input_token(self, query, history):
        system_context = "You are a helpful assistant."
        max_window_size = 1536
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system_context)
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
            + self.tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
        return torch.tensor([context_tokens]).to("cpu")

    def config(self):
        return {
            "do_sample": False,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            # "<|im_end|>":151645, "\n": 198, "<|im_start|>":151644
            "stop_words_ids": [[151645, 198, 151644]]
        }

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">通义千问</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = QwenDemo(args.token_path, args.model_path, dtype=args.dtype)

    demo.launch(False)
