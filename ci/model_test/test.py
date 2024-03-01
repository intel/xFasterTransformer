import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import torch
from transformers import AutoTokenizer
import json

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config json file")

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Config file is not exist.")
        exit(-1)

    with open(args.config, "r") as json_file:
        config = json.load(json_file)

    name = config["name"] if "name" in config.keys() else "XFT"
    token_path = config["token_path"]
    model_path = config["model_path"]
    data_types = config["data_types"]
    test_case = config["test_case"]

    print(f"[INFO] Model name: {name}")
    print(f"[INFO] Tokenizer path: {token_path}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Data type list: {data_types}")
    print(f"[INFO] Searcher type list: {list(test_case.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

    success_count = 0
    fail_count = 0

    for dtype in data_types:
        model = xfastertransformer.AutoModel.from_pretrained(model_path, dtype=dtype)
        for searcher in test_case.keys():
            num_beams = test_case[searcher]["num_beams"]
            for case_name in test_case[searcher]["case"]:
                case = test_case[searcher]["case"][case_name]
                input_prompt = case["input"]
                output_len = case["output_len"]
                expected_output = case["expected_output"]
                input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
                expected_output_ids = tokenizer(expected_output, return_tensors="pt").input_ids

                generated_ids = model.generate(
                    input_ids, max_length=input_ids.shape[-1] + output_len, num_beams=num_beams
                )
                output_ids = generated_ids[..., -1 * output_len :]

                ret = tokenizer.decode(torch.squeeze(output_ids), skip_special_tokens=True)
                output_ids = tokenizer(ret, return_tensors="pt").input_ids

                if model.rank != 0:
                    continue
                if torch.equal(expected_output_ids, output_ids):
                    success_count += 1
                else:
                    print("[INFO][Failed] ====================")
                    print(f"[INFO][Failed] Searcher type: {searcher}")
                    print(f"[INFO][Failed] Num_beams: {num_beams}")
                    print(f"[INFO][Failed] Case Name: {case_name}")
                    print(f"[INFO][Failed] Input prompt: {input_prompt}")
                    print(f"[INFO][Failed] Data type: {dtype}")
                    print(f"[INFO][Failed] Expect output ids: {expected_output_ids}")
                    print(f"[INFO][Failed] Expect output: {expected_output}")
                    print(f"[INFO][Failed] Output ids: {output_ids}")
                    print(f"[INFO][Failed] Output: {ret}")
                    fail_count += 1

    if success_count != 0 or fail_count != 0:
        print(f"[INFO] Succeed : {success_count}, Failed : {fail_count}.")
