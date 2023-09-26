import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import argparse
import time
import torch
from demo_utils import ChatDemo


parser = argparse.ArgumentParser()
parser.add_argument("--token_path", type=str, default="/data/Llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("--model_path", type=str, default="/data/Llama-2-7b-chat-cpu", help="Path to model file")
parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "int8"], default="fp16", help="Data type")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
EOS_ID = 2


class Llama2Demo(ChatDemo):
    def post_process_generation(self, next_token_id, token_list, chatbot, query, history):
        token_list.extend(next_token_id)
        response = self.tokenizer.decode(token_list, skip_special_tokens=True)
        new_history = history + [(query, response)]
        chatbot[-1] = (self.parse_text(query), self.parse_text(response))
        return chatbot, new_history

    # Refer to https://github.com/facebookresearch/llama/blob/main/llama/generation.py
    def predict(self, query, chatbot, max_length, history):
        chatbot.append((self.parse_text(query), ""))
        self.model.config(max_length)
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

        self.model.input(input_tokens)
        token_list = []
        time_cost = []

        while not self.model.is_done():
            start_time = time.perf_counter()
            next_token_id = self.model.forward().view(-1).tolist()
            end_time = time.perf_counter()
            time_cost.append(end_time - start_time)
            yield self.post_process_generation(next_token_id, token_list, chatbot, query, history)

        total_cost = sum(time_cost[1:])
        latency = total_cost * 1000 / len(time_cost[1:])
        throughput = len(time_cost[1:]) / total_cost
        response = self.tokenizer.decode(token_list, skip_special_tokens=True)
        print(f"Query is : {query.strip()}")
        print(f"Response is : {response}")
        print(f"Latency:\t{latency:.2f} ms/token")
        print(f"Througput:\t{throughput:.2f} tokens/s")

    def html_func(self):
        gr.HTML("""<h1 align="center">xFasterTransformer</h1>""")
        gr.HTML("""<h1 align="center">Llama2</h1>""")


if __name__ == "__main__":
    args = parser.parse_args()
    demo = Llama2Demo(args.token_path, args.model_path, dtype=args.dtype)

    demo.launch(False)
