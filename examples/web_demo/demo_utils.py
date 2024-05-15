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
import time
import gradio as gr
import importlib.util
from transformers import AutoTokenizer
import transformers
import json
import os
import sys
import traceback

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer


def clean_input():
    return gr.update(value="")


def reset():
    return [], []


XFT_DTYPE_LIST = [
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

XFT_KVCACHE_DTYPE_LIST = ["fp16", "int8"]


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


class ChatDemo:
    def __init__(self, token_path, model_path, dtype, kv_cache_dtype: str = "fp16"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        except Exception as e:
            traceback.print_exc()
            print("[ERROR] An exception occurred during the tokenizer loading process.")
            # print(f"{type(e).__name__}: {str(e)}")
            check_transformers_version_compatibility(token_path)
            sys.exit(-1)

        self.model = xfastertransformer.AutoModel.from_pretrained(
            model_path, dtype=dtype, kv_cache_dtype=kv_cache_dtype
        )

    @property
    def rank(self):
        return self.model.rank()

    def predict(self, query, chatbot, model, max_length, history):
        pass

    def html_func(self):
        pass

    def launch(self, share=False):
        # Master
        if self.model.rank == 0:
            with gr.Blocks() as demo:
                self.html_func()

                chatbot = gr.Chatbot()
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Column(scale=12):
                            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=7, container=False)
                        with gr.Column(min_width=32, scale=1):
                            submitBtn = gr.Button("Submit", variant="primary")
                    with gr.Column(scale=1):
                        perf_txt = gr.Textbox(
                            value="Latency:\t0 ms\nThrougput:\t0 tokens/s",
                            interactive=False,
                            container=False,
                            lines=3,
                        )
                        batch_size = gr.Slider(1, 8, value=1, step=1.0, label="Batch size", interactive=True)
                        emptyBtn = gr.Button("Clear History")
                        max_length = gr.Slider(
                            0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True, visible=False
                        )

                history = gr.State([])
                submitBtn.click(
                    self.predict,
                    [user_input, chatbot, batch_size, max_length, history],
                    [chatbot, history, perf_txt],
                    show_progress=True,
                )
                submitBtn.click(clean_input, [], [user_input])
                emptyBtn.click(reset, [], [chatbot, history], show_progress=True)

            demo.queue().launch(share=share, inbrowser=True)
        else:
            # Slave
            while True:
                self.model.generate()

    def parse_text(self, text):
        return text

    def process_response(self, response):
        return response

    def config(self):
        return {}

    def post_process_generation(self, next_token_id, token_list, chatbot, query, history, perf_info):
        token_list.extend(next_token_id)
        response = self.tokenizer.decode(token_list, skip_special_tokens=True)
        response = self.process_response(response)
        new_history = history + [(query, response)]
        chatbot[-1] = (self.parse_text(query), self.parse_text(response))
        return chatbot, new_history, perf_info

    def create_chat_input_token(self, query, history):
        input_tokens = self.tokenizer([query], return_tensors="pt").input_ids
        return input_tokens

    def predict(self, query, chatbot, batch_size, max_length, history):
        chatbot.append((self.parse_text(query), ""))
        self.model.config(max_length, **self.config())

        input_tokens = self.create_chat_input_token(query, history)
        if batch_size > 1:
            input_tokens = input_tokens.tile((batch_size, 1))

        self.model.input(input_tokens)
        token_list = []
        time_cost = []

        while not self.model.is_done():
            start_time = time.perf_counter()
            next_token_id = self.model.forward()
            end_time = time.perf_counter()
            next_token_id = next_token_id.view(-1).tolist()[:1]
            time_cost.append(end_time - start_time)
            perf_info = ""
            if len(time_cost) > 1:
                total_cost = sum(time_cost[1:])
                latency = total_cost * 1000 / len(time_cost[1:])
                throughput = (len(time_cost[1:]) * batch_size) / total_cost
                perf_info = f"Latency:\t{latency:.2f} ms\nThrougput:\t{throughput:.2f} tokens/s"
            yield self.post_process_generation(next_token_id, token_list, chatbot, query, history, perf_info)

        response = self.tokenizer.decode(token_list, skip_special_tokens=True)
        response = self.process_response(response)
        print(f"Query is : {query.strip()}")
        print(f"Response is : {response}")
        if len(time_cost) > 1:
            total_cost = sum(time_cost[1:])
            latency = total_cost * 1000 / len(time_cost[1:])
            throughput = (len(time_cost[1:]) * batch_size) / total_cost
            print(f"Latency:\t{latency:.2f} ms")
            print(f"Througput:\t{throughput:.2f} tokens/s")
