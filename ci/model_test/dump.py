import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import torch
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, OPTForCausalLM
import json


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config json file")
parser.add_argument("--dump_file", type=str, required=True, help="Path to dumped config json file")


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Config file is not exist.")
        exit(-1)

    if not os.path.exists(args.dump_file):
        print("Config file is not exist.")
        exit(-1)

    with open(args.config, "r") as json_file:
        config = json.load(json_file)

    name = config["name"].lower()
    token_path = config["token_path"]

    tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
    if "llama" in name:
        model = LlamaForCausalLM.from_pretrained(token_path)
    elif "opt" in name:
        model = OPTForCausalLM.from_pretrained(token_path)
    else:
        model = AutoModel.from_pretrained(token_path, trust_remote_code=True)
    if "chatglm" in name:
        model = model.to("cpu", dtype=float)

    for searcher in config["test_case"].keys():
        num_beams = config["test_case"][searcher]["num_beams"]
        for case_name in config["test_case"][searcher]["case"]:
            input_prompt = config["test_case"][searcher]["case"][case_name]["input"]
            output_len = config["test_case"][searcher]["case"][case_name]["output_len"]
            input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            generated_ids = model.generate(input_ids, max_length=input_ids.shape[-1] + output_len, num_beams=num_beams)
            output_ids = generated_ids[..., -1 * output_len :]
            ret = tokenizer.decode(torch.squeeze(output_ids), skip_special_tokens=True)
            config["test_case"][searcher]["case"][case_name]["expected_output"] = ret

    with open(args.dump_file, "w", encoding="utf8") as file:
        json.dump(config, file, ensure_ascii=False)
