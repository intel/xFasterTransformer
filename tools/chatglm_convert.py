"""
Convert huggingface ChatGLM model. Use https://huggingface.co/THUDM/chatglm-6b
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
from transformers import AutoTokenizer, AutoModel
from convert_tools import compatibility_check

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../../.."))
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
        path = os.path.join(saved_dir, "model." + key)
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

    elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    elif "mlp.dense_h_to_4h.weight" in key or "mlp.dense_h_to_4h.bias" in key:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    elif "attention.query_key_value.bias" in key:
        hidden_dim = (int)(val.shape[-1] / 3)

        # TODO: hard coded num_attention_heads as 32, size per head as 128
        val = val.reshape(32, 128 * 3)
        qkv = np.split(val, 3, axis=-1)
        q = qkv[0].reshape(1, hidden_dim)
        k = qkv[1].reshape(1, hidden_dim)
        v = qkv[2].reshape(1, hidden_dim)
        val = np.concatenate((q, k, v), axis=-1)

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            save_val(split_vals[j], key, i * factor + j)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0]

        # TODO: hard coded num_attention_heads as 32, size per head as 128
        val = val.reshape(hidden_dim, 32, 128 * 3)
        qkv = np.split(val, 3, axis=-1)
        q = qkv[0].reshape(hidden_dim, hidden_dim)
        k = qkv[1].reshape(hidden_dim, hidden_dim)
        v = qkv[2].reshape(hidden_dim, hidden_dim)
        val = np.concatenate((q, k, v), axis=-1)

        split_vals = np.split(val, factor, axis=-1)
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
    model = AutoModel.from_pretrained(args.in_file, trust_remote_code=True)

    hf_config = vars(model.config)

    layer_names = [name for name, param in model.named_parameters()]

    # save parameters to config file
    config = configparser.ConfigParser()
    config["chatglm"] = {}
    has_post_decoder_layernorm = "model.decoder.final_layer_norm.bias" in layer_names
    try:
        config["chatglm"]["model_name"] = "chatglm" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
        config["chatglm"]["head_num"] = str(hf_config["num_attention_heads"])
        n_embd = hf_config["hidden_size"]
        config["chatglm"]["size_per_head"] = str(n_embd // hf_config["num_attention_heads"])
        config["chatglm"]["inter_size"] = str(hf_config["inner_hidden_size"])
        config["chatglm"]["max_pos_seq_len"] = str(hf_config["max_sequence_length"])
        config["chatglm"]["num_layer"] = str(hf_config["num_layers"])
        config["chatglm"]["layernorm_eps"] = "1e-5"
        config["chatglm"]["layernorm_type"] = "pre_layernorm"
        config["chatglm"]["activation_type"] = "Gelu"
        config["chatglm"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
        config["chatglm"]["vocab_size"] = str(hf_config["vocab_size"])
        config["chatglm"]["start_id"] = str(hf_config["bos_token_id"])
        config["chatglm"]["end_id"] = str(hf_config["eos_token_id"])
        config["chatglm"]["weight_data_type"] = args.weight_data_type
        with open(os.path.join(saved_dir, "config.ini"), "w") as configfile:
            config.write(configfile)
    except Exception as e:
        print("Fail to save the config in config.ini.", str(e))

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    huggingface_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    state_dict = model.state_dict()
    model_named_parameters = dict()
    for name, param in state_dict.items():
        print(name)
        if "embed" in name:
            model_named_parameters[name] = param
        elif "lm_head" in name:
            model_named_parameters[name] = param
        else:
            model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

    pool = multiprocessing.Pool(args.processes)
    for name, param in model_named_parameters.items():
        if name == "transformer.word_embeddings.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(os.path.join(saved_dir, "model.wte.bin"))
        elif name == "transformer.final_layernorm.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                os.path.join(saved_dir, "model.final_layernorm.weight.bin")
            )
        elif name == "transformer.final_layernorm.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                os.path.join(saved_dir, "model.final_layernorm.bias.bin")
            )
        elif name == "lm_head.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                os.path.join(saved_dir, "model.lm_head.weight.bin")
            )
        else:
            starmap_args = []
            for i in range(len(huggingface_model_name_pattern)):
                if huggingface_model_name_pattern[i] in name:
                    factor = 1
                    new_name = name.replace("transformer.layers.", "layers.").replace(
                        huggingface_model_name_pattern[i], ft_model_name_pattern[i]
                    )
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
    parser.add_argument("--saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("--in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("--processes", "-p", type=int, help="processes to spawn for conversion (default: 8)", default=8)
    parser.add_argument("--weight_data_type", "-d", type=str, default="fp16", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    compatibility_check(args.in_file)

    start_time = datetime.now()
    split_and_convert(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
