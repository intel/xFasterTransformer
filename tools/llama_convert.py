"""
Convert huggingface ChatGLM model. Use https://huggingface.co/meta-llama
"""

import argparse
import configparser
import multiprocessing
import numpy as np
import os
import sys
import torch

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(i, saved_dir, factor, key, args, val, old_name, dtype):
    def save_val(val, key, tp_num=None):
        if key.startswith("model."):
            path = saved_dir + "/" + key
        else:
            path = saved_dir + "/model." + key

        if tp_num is not None:
            path += "." + str(tp_num)
        path += ".bin"

        val.tofile(path)

    if (
        "input_layernorm.weight" in key
        or "input_layernorm.bias" in key
        or "attention.dense.bias" in key
        or "post_attention_layernorm.weight" in key
        or "post_attention_layernorm.bias" in key
        or "mlp.dense_4h_to_h.bias" in key
        or "final_layernorm.weight" in key
        or "final_layernorm.bias" in key
    ):
        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            save_val(val, key)

    elif "mlp.gate_proj.weight" in key or "mlp.up_proj.weight" in key or "mlp.down_proj.weight" in key:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0]
        local_dim = (int)(val.shape[-1] / 3)

        val = val.reshape(hidden_dim, 3, local_dim)

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    elif "attention.dense.weight" in key:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    else:
        print("[ERROR] cannot find key '{}'".format(key))


def split_and_convert(args):
    saved_dir = args.saved_dir

    # create directory if not exist
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # load the model
    model = LlamaForCausalLM.from_pretrained(
        args.in_file,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    hf_config = vars(model.config)

    layer_names = [name for name, param in model.named_parameters()]

    # save parameters to config file
    config = configparser.ConfigParser()
    config["llama"] = {}
    has_post_decoder_layernorm = True
    try:
        config["llama"]["model_name"] = "llama" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
        config["llama"]["head_num"] = str(hf_config["num_attention_heads"])
        hidden_size = hf_config["hidden_size"]
        config["llama"]["size_per_head"] = str(hidden_size // hf_config["num_attention_heads"])
        config["llama"]["inter_size"] = str(hf_config["intermediate_size"])
        config["llama"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
        config["llama"]["num_layer"] = str(hf_config["num_hidden_layers"])
        config["llama"]["rms_norm_eps"] = "1e-6"
        config["llama"]["layernorm_type"] = "pre_layernorm"
        config["llama"]["activation_type"] = "silu"
        config["llama"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
        config["llama"]["vocab_size"] = str(hf_config["vocab_size"])
        config["llama"]["start_id"] = str(hf_config["bos_token_id"])
        config["llama"]["end_id"] = str(hf_config["eos_token_id"])
        config["llama"]["weight_data_type"] = args.weight_data_type
        with open(saved_dir + "/config.ini", "w") as configfile:
            config.write(configfile)
    except Exception as e:
        print("Fail to save the config in config.ini.", str(e))

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    hf_model_name_pattern = [
        "input_layernorm.weight",
        "attention.query_key_value.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]

    ft_model_name_pattern = [
        "input_layernorm.weight",
        "attention.query_key_value.weight",
        "attention.dense.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]

    state_dict = model.state_dict()
    model_named_parameters = dict()
    for name, param in state_dict.items():
        print(name)
        # merge QKV
        if "self_attn.q_proj.weight" in name:
            k_name = name.replace("q_proj", "k_proj")
            v_name = name.replace("q_proj", "v_proj")
            qkv = torch.cat(
                (param.permute(1, 0), state_dict[k_name].permute(1, 0), state_dict[v_name].permute(1, 0)), dim=1
            )
            model_named_parameters[name.replace("self_attn.q_proj.weight", "attention.query_key_value.weight")] = qkv
        # for merged weights, skip
        elif "self_attn.k_proj.weight" in name or "self_attn.v_proj.weight" in name:
            continue
        elif "embed" in name:
            model_named_parameters[name] = param
        elif "lm_head" in name:
            model_named_parameters[name] = param
        else:
            model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

    pool = multiprocessing.Pool(args.processes)
    for name, param in model_named_parameters.items():
        if name == "model.embed_tokens.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.wte.bin")
        elif name == "model.norm.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.final_layernorm.weight.bin"
            )
        # elif name == 'model.final_layernorm.bias':
        #     param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
        #         saved_dir + "model.final_layernorm.bias.bin")
        elif name == "lm_head.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.lm_head.weight.bin")
        else:
            starmap_args = []
            for i in range(len(hf_model_name_pattern)):
                if hf_model_name_pattern[i] in name:
                    factor = 1
                    new_name = name.replace(hf_model_name_pattern[i], ft_model_name_pattern[i])
                    starmap_args.append(
                        (
                            0,
                            saved_dir,
                            factor,
                            new_name,
                            args,
                            param.detach().cpu().numpy().astype(np_weight_data_type),
                            name,
                            np_weight_data_type,
                        )
                    )
            pool.starmap_async(split_and_convert_process, starmap_args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-processes", "-p", type=int, help="processes to spawn for conversion (default: 8)", default=8)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.now()
    split_and_convert(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
