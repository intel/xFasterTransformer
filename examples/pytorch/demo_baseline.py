import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer, TextStreamer, AutoModel

Token_path = "/data/opt-13b"
Model_path = "/data/opt-13b"

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(Token_path, use_fast=False, padding_side="left", trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False)
    model = AutoModel.from_pretrained(Model_path, trust_remote_code=True)

    while True:
        input_prompt = input("\nPlease enter the prompt: ")
        if input_prompt == "":
            input_prompt = DEFAULT_PROMPT
            print("[Use default prompt]:" + input_prompt)
        input_ids = tokenizer(input_prompt, return_tensors="pt", padding=True).input_ids
        print("=" * 50)

        start_time = time.perf_counter()
        generated_ids = model.generate(input_ids, max_length=193, streamer=streamer)
        end_time = time.perf_counter()

        if streamer is None:
            ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for snt in ret:
                print(snt)
        print("=" * 20 + "Performance" + "=" * 20)
        execution_time = end_time - start_time
        print(f"Execution time:\t{execution_time:.2f} s")
        input_token_nums = torch.numel(input_ids)
        output_token_nums = torch.numel(generated_ids) - input_token_nums
        latency = execution_time * 1000 / output_token_nums
        througput = output_token_nums / execution_time
        print(f"Latency:\t{latency:.2f} ms/token")
        print(f"Througput:\t{througput:.2f} tokens/s")
