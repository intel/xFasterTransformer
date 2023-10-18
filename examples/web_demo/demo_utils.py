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
import mdtex2html
import re
import gradio as gr
import importlib.util
from transformers import AutoTokenizer

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


# Override Chatbot.postprocess
gr.Chatbot.postprocess = postprocess


def clean_input():
    return gr.update(value="")


def reset():
    return [], []


class ChatDemo:
    def __init__(self, token_path, model_path, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        self.model = xfastertransformer.AutoModel.from_pretrained(model_path, dtype=dtype)

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
                    with gr.Column(scale=4):
                        with gr.Column(scale=12):
                            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(
                                container=False
                            )
                        with gr.Column(min_width=32, scale=1):
                            submitBtn = gr.Button("Submit", variant="primary")
                    with gr.Column(scale=1):
                        emptyBtn = gr.Button("Clear History")
                        max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)

                history = gr.State([])
                submitBtn.click(
                    self.predict, [user_input, chatbot, max_length, history], [chatbot, history], show_progress=True
                )
                submitBtn.click(clean_input, [], [user_input])
                emptyBtn.click(reset, [], [chatbot, history], show_progress=True)

            demo.queue().launch(share=share, inbrowser=True)
        else:
            # Slave
            while True:
                self.model.generate()

    # Process code blocks with HTML tags, and escape certain special characters
    def parse_text(self, text):
        text = re.compile(r"(?<!\n)```|```(?!`\n)").sub("\n```\n", text)
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        puncts_map = {
            "`": "\`",
            "<": "&lt;",
            ">": "&gt;",
            " ": "&nbsp;",
            "*": "&ast;",
            "_": "&lowbar;",
            "-": "&#45;",
            ".": "&#46;",
            "!": "&#33;",
            "(": "&#40;",
            ")": "&#41;",
            "$": "&#36;",
        }
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
                        for key, val in puncts_map.items():
                            line = line.replace(key, val)
                    lines[i] = "<br>" + line
        return "".join(lines)

    # Replace English punctuation with Chinese punctuation in the Chinese sentences.
    def process_response(self, response):
        response = response.strip()
        puncts = {",": "，", "!": "！", ":": "：", ";": "；", "\?": "？"}
        for eng, chn in puncts.items():
            response = re.sub(r"([\u4e00-\u9fff])%s" % eng, r"\1%s" % chn, response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % eng, r"%s\1" % chn, response)
        return response
