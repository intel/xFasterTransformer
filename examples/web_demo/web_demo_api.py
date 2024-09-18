import gradio as gr
import argparse

from openai import OpenAI


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", type=str, default="http://local:8000/v1", help="base url")
parser.add_argument("-m", "--model", type=str, default="xft", help="model name")
parser.add_argument("-t", "--token", type=str, default="EMPTY", help="api key")
parser.add_argument("-i", "--ip", type=str, default="0.0.0.0", help="gradio server ip")
parser.add_argument("-p", "--port", type=int, default=7860, help="gradio server port")
parser.add_argument("-s", "--share", type=boolean_string, default=False, help="create a share link")


def clean_input():
    return gr.update(value="")


def reset():
    return [], []


class ChatDemo:
    def __init__(self, url, model, token):
        self.client = OpenAI(
            base_url=url,
            api_key=token,
        )
        self.model = model

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")

    def launch(self, server_name="0.0.0.0", server_port=7860, share=False):
        with gr.Blocks() as demo:
            self.html_func()

            chatbot = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=1, container=False)
                with gr.Column(scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=1):
                    emptyBtn = gr.Button("Clear History")

            history = gr.State([])
            submitBtn.click(
                self.predict,
                [user_input, chatbot, history],
                [chatbot, history],
                show_progress=True,
            )
            submitBtn.click(clean_input, [], [user_input])
            emptyBtn.click(reset, [], [chatbot, history], show_progress=True)

        demo.queue().launch(server_name=server_name, server_port=server_port, share=share, inbrowser=True)

    def post_process_generation(self, chunk, chatbot, query, history):
        response = chatbot[-1][1] + chunk
        new_history = history + [(query, response)]
        chatbot[-1] = (query, response)
        return chatbot, new_history

    def create_chat_input(self, query, history):
        msgs = []
        if history is None:
            history = []
        for user_msg, model_msg in history:
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": model_msg})
        msgs.append({"role": "user", "content": query})
        return msgs

    def predict(self, query, chatbot, history):
        chatbot.append((query, ""))

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.create_chat_input(query, history),
            max_tokens=2048,
            stream=True,
            temperature=1.0,
            extra_body={"top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1},
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield self.post_process_generation(chunk.choices[0].delta.content, chatbot, query, history)


if __name__ == "__main__":
    args = parser.parse_args()
    demo = ChatDemo(args.url, args.model, args.token)

    demo.launch(args.ip, args.port, args.share)
