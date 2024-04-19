# Copyright (c) 2023-2024 Intel Corporation
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

import configparser
import multiprocessing
import numpy as np
import os
import torch

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation import GenerationConfig

from .convert import BaseModelConvert, get_name_and_param


class Qwen2Convert(BaseModelConvert):
    """
    Convert Qwen model. Use https://huggingface.co/Qwen or https://modelscope.cn/models
    """

    def __init__(self):
        super().__init__()

    def split_and_convert_process(self, i, saved_dir, factor, key, val, num_attention_heads, num_key_value_heads):
        def save_val(val, key, tp_num=None):
            if key.startswith("model."):
                path = os.path.join(saved_dir, key)
            else:
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

        elif "mlp.gate_proj.weight" in key or "mlp.up_proj.weight" in key or "mlp.down_proj.weight" in key:
            split_vals = np.split(val, factor, axis=0)
            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        elif "attention.query_key_value.weight" in key:
            qkvcols = val.shape[-1]
            head_size = int(qkvcols / (int(num_attention_heads) + int(num_key_value_heads) * 2))
            qcol = int(num_attention_heads) * head_size
            kcol = int(num_key_value_heads) * head_size
            vcol = int(num_key_value_heads) * head_size
            qkv = np.split(val, [qcol, (qcol + kcol)], axis=-1)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            q_split_vals = np.split(q, factor, axis=-1)
            k_split_vals = np.split(k, factor, axis=-1)
            v_split_vals = np.split(v, factor, axis=-1)
            for j in range(factor):
                val = np.concatenate((q_split_vals[j], k_split_vals[j], v_split_vals[j]), axis=-1)
                save_val(val, key, i * factor + j)

        elif "attention.query_key_value.bias" in key:
            hidden_dim = val.shape[0]
            head_size = int(hidden_dim / (int(num_attention_heads) + int(num_key_value_heads) * 2))
            qcol = int(num_attention_heads) * head_size
            kcol = int(num_key_value_heads) * head_size
            vcol = int(num_key_value_heads) * head_size
            qkv = np.split(val, [qcol, qcol + kcol])
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            q_split_vals = np.split(q, factor, axis=-1)
            k_split_vals = np.split(k, factor, axis=-1)
            v_split_vals = np.split(v, factor, axis=-1)
            for j in range(factor):
                val = np.concatenate((q_split_vals[j], k_split_vals[j], v_split_vals[j]), axis=-1)
                save_val(val, key, i * factor + j)

        elif "attention.dense.weight" in key:
            split_vals = np.split(val, factor, axis=0)
            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        else:
            print("[ERROR] cannot find key '{}'".format(key))

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        saved_dir = output_dir

        # create directory if not exist
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        # load the model
        gen_config = GenerationConfig.from_pretrained(input_dir, trust_remote_code=True, resume_download=True)
        hf_config, _ = AutoConfig.from_pretrained(
            input_dir, return_unused_kwargs=True, trust_remote_code=True, fp16=True, use_flash_attn=False
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        hf_config = {
            **vars(gen_config),
            **vars(hf_config),
        }

        sec_name = hf_config["model_type"]

        # save parameters to config file
        config = configparser.ConfigParser()
        config[sec_name] = {}
        has_post_decoder_layernorm = True
        try:
            config[sec_name]["model_name"] = "qwen2" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            num_attention_heads = config[sec_name]["head_num"] = str(hf_config["num_attention_heads"])
            num_key_value_heads = config[sec_name]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            hidden_size = hf_config["hidden_size"]
            config[sec_name]["size_per_head"] = str(hidden_size // hf_config["num_attention_heads"])
            config[sec_name]["inter_size"] = str(hf_config["intermediate_size"])
            config[sec_name]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config[sec_name]["num_layer"] = str(hf_config["num_hidden_layers"])
            config[sec_name]["rms_norm_eps"] = str(hf_config["rms_norm_eps"])
            config[sec_name]["layernorm_type"] = "pre_layernorm"
            config[sec_name]["activation_type"] = str(hf_config["hidden_act"])
            config[sec_name]["rope_theta"] = str(hf_config["rope_theta"])
            config[sec_name]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config[sec_name]["vocab_size"] = str(hf_config["vocab_size"])
            config[sec_name]["start_id"] = str(hf_config["bos_token_id"])
            config[sec_name]["end_id"] = str(hf_config["eos_token_id"])
            config[sec_name]["pad_id"] = str(gen_config.pad_token_id)
            config[sec_name]["weight_data_type"] = dtype
            with open(os.path.join(saved_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))
            exit(-1)

        hf_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "attention.query_key_value.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]

        ft_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "attention.query_key_value.bias",
            "attention.dense.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]

        print("Processing ...")
        state_dict = model.state_dict()
        model_named_parameters = dict()
        for name, param in state_dict.items():
            # print(f"name = {name}")
            # merge QKV
            if "self_attn.q_proj.weight" in name:
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                qkv = torch.cat(
                    (param.permute(1, 0), state_dict[k_name].permute(1, 0), state_dict[v_name].permute(1, 0)), dim=1
                )
                model_named_parameters[name.replace("self_attn.q_proj.weight", "attention.query_key_value.weight")] = (
                    qkv
                )
            # for merged weights, skip
            elif "self_attn.k_proj.weight" in name or "self_attn.v_proj.weight" in name:
                continue
            # merge QKV bias
            if "self_attn.q_proj.bias" in name:
                print(f"name = {name}")
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                qkv_bias = torch.cat((param, state_dict[k_name], state_dict[v_name]))
                model_named_parameters[name.replace("self_attn.q_proj.bias", "attention.query_key_value.bias")] = (
                    qkv_bias
                )
            # for merged weights, skip
            elif "self_attn.k_proj.bias" in name or "self_attn.v_proj.bias" in name:
                continue
            elif "embed" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            elif "gemma" in sec_name and "norm" in name:
                model_named_parameters[name] = param + 1
            elif "layernorm" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        pool = multiprocessing.Pool(processes)
        for name, param in model_named_parameters.items():
            param = param.half()
            if name == "model.embed_tokens.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(os.path.join(output_dir, "model.wte.bin"))
            elif name == "model.norm.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
                )
            elif name == "lm_head.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(saved_dir, "model.lm_head.weight.bin")
                )
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
                                param.detach().cpu().numpy().astype(self.dtype),
                                num_attention_heads,
                                num_key_value_heads,
                            )
                        )
                pool.starmap_async(self.split_and_convert_process, starmap_args)
        pool.close()
        pool.join()

        print(f"{saved_dir} export successful!")
