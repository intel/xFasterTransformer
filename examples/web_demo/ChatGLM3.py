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
from demo_utils import ChatDemo, XFT_DTYPE_LIST, XFT_KVCACHE_DTYPE_LIST


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/chatglm3-6b-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/chatglm3-6b-xft", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=XFT_DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=XFT_KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")


class ChatGLM3Demo(ChatDemo):
    # Replace English punctuation with Chinese punctuation in the Chinese sentences.
    def process_response(self, output):
        content = ""
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                content = content.replace("[[训练时间]]", "2023年")
        return content

    def create_chat_input_token(self, query, history):
        if history is None:
            history = []
        input_ids = []
        for old_query, response in history:
            input_ids.extend(self.tokenizer.build_single_message("user", "", old_query))
            input_ids.extend(self.tokenizer.build_single_message("assistant", "", response))
        input_ids.extend(self.tokenizer.build_single_message("user", "", query))
        input_ids.extend([self.tokenizer.get_command("<|assistant|>")])
        inputs = self.tokenizer.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True).input_ids
        return inputs

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">ChatGLM3</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = ChatGLM3Demo(args.token_path, args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype)

    demo.launch(False)
