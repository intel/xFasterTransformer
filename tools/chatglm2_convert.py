"""
Convert huggingface ChatGLM model. Use https://huggingface.co/THUDM/chatglm2-6b
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


def split_and_convert_process(
    i, saved_dir, factor, key, args, val, old_name, dtype, num_attention_heads, multi_query_group_num, kv_channels
):
    def save_val(val, key, tp_num=None):
        path = path = os.path.join(saved_dir, "model." + key)
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
        hidden_dim = val.shape[0]
        qcol = int(num_attention_heads)
        kcol = int(multi_query_group_num)
        vcol = int(multi_query_group_num)
        kv_channels = int(kv_channels)
        qkv = np.split(val, [qcol, qcol + kcol])
        q = qkv[0]  # .reshape(1, qcol * kv_channels)
        k = qkv[1]  # .reshape(1, kcol * kv_channels)
        v = qkv[2]  # .reshape(1, vcol * kv_channels)
        q_split_vals = np.split(q, factor, axis=-1)
        k_split_vals = np.split(k, factor, axis=-1)
        v_split_vals = np.split(v, factor, axis=-1)
        for j in range(factor):
            val = np.concatenate((q_split_vals[j], k_split_vals[j], v_split_vals[j]), axis=-1)
            save_val(val, key, i * factor + j)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0]
        kv_channels = int(kv_channels)
        qcol = int(num_attention_heads) * kv_channels
        kcol = int(multi_query_group_num) * kv_channels
        vcol = int(multi_query_group_num) * kv_channels
        # val = val.reshape(hidden_dim, kv_channels, (qcol + kcol + vcol))
        qkv = np.split(val, [qcol, (qcol + kcol)], axis=-1)
        q = qkv[0]  # .reshape(hidden_dim, qcol * kv_channels)
        k = qkv[1]  # .reshape(hidden_dim, kcol * kv_channels)
        v = qkv[2]  # .reshape(hidden_dim, vcol * kv_channels)
        # val = np.concatenate((q, k, v), axis=-1)
        q_split_vals = np.split(q, factor, axis=-1)
        k_split_vals = np.split(k, factor, axis=-1)
        v_split_vals = np.split(v, factor, axis=-1)
        for j in range(factor):
            val = np.concatenate((q_split_vals[j], k_split_vals[j], v_split_vals[j]), axis=-1)
            save_val(val, key, i * factor + j)

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
    print("hf_config= {}".format(hf_config))

    layer_names = [name for name, param in model.named_parameters()]

    # save parameters to config file
    config = configparser.ConfigParser()
    config["chatglm2"] = {}
    has_post_decoder_layernorm = "model.decoder.final_layer_norm.bias" in layer_names
    try:
        config["chatglm2"]["model_name"] = (
            "chatglm2" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
        )
        num_attention_heads = config["chatglm2"]["head_num"] = str(hf_config["num_attention_heads"])
        n_embd = hf_config["hidden_size"]
        config["chatglm2"]["size_per_head"] = str(n_embd // hf_config["num_attention_heads"])
        config["chatglm2"]["inter_size"] = str(hf_config["ffn_hidden_size"])
        config["chatglm2"]["max_pos_seq_len"] = str(hf_config["seq_length"])
        config["chatglm2"]["num_layer"] = str(hf_config["num_layers"])
        config["chatglm2"]["layernorm_eps"] = str(hf_config["layernorm_epsilon"])  # "1e-5"
        config["chatglm2"]["layernorm_type"] = "pre_layernorm"
        config["chatglm2"]["activation_type"] = "swiglu"
        config["chatglm2"]["has_post_decoder_layernorm"] = "1" if str(hf_config["post_layer_norm"]) == "True" else "0"
        config["chatglm2"]["vocab_size"] = str(hf_config["padded_vocab_size"])
        config["chatglm2"]["start_id"] = str(hf_config["bos_token_id"])
        config["chatglm2"]["end_id"] = str(hf_config["eos_token_id"])
        config["chatglm2"]["weight_data_type"] = args.weight_data_type

        kv_channels = config["chatglm2"]["kv_channels"] = str(hf_config["kv_channels"])
        config["chatglm2"]["rmsnorm"] = "1" if str(hf_config["rmsnorm"]) == "True" else "0"
        config["chatglm2"]["apply_residual_connection_post_layernorm"] = (
            "1" if str(hf_config["apply_residual_connection_post_layernorm"]) == "True" else "0"
        )
        config["chatglm2"]["multi_query_attention"] = "1" if str(hf_config["multi_query_attention"]) == "True" else "0"
        multi_query_group_num = config["chatglm2"]["kv_head_num"] = str(hf_config["multi_query_group_num"])
        config["chatglm2"]["pad_id"] = str(hf_config["pad_token_id"])

        with open(os.path.join(saved_dir, "config.ini"), "w") as configfile:
            config.write(configfile)
    except Exception as e:
        print("Fail to save the config in config.ini.", str(e))

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    huggingface_model_name_pattern = [
        "input_layernorm.bias",  #
        "input_layernorm.weight",
        "self_attention.query_key_value.bias",
        "self_attention.query_key_value.weight",
        "self_attention.dense.bias",  #
        "self_attention.dense.weight",
        "post_attention_layernorm.bias",  #
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",  #
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",  #
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
        if "embed" in name:
            model_named_parameters[name] = param
        elif "output_layer" in name:
            model_named_parameters[name] = param
        else:
            model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

    pool = multiprocessing.Pool(args.processes)
    for name, param in model_named_parameters.items():
        if name == "transformer.embedding.word_embeddings.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(os.path.join(saved_dir, "model.wte.bin"))
        elif name == "transformer.encoder.final_layernorm.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                os.path.join(saved_dir, "model.final_layernorm.weight.bin")
            )
        elif name == "transformer.output_layer.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                os.path.join(saved_dir, "model.lm_head.weight.bin")
            )
        else:
            starmap_args = []
            for i in range(len(huggingface_model_name_pattern)):
                if huggingface_model_name_pattern[i] in name:
                    factor = 1
                    new_name = name.replace("transformer.encoder.layers.", "layers.").replace(
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
                            num_attention_heads,
                            multi_query_group_num,
                            kv_channels,
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
