# Copyright (c) 2025 Intel Corporation
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
from tqdm import tqdm
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation import GenerationConfig

from .convert import BaseModelConvert, get_name_and_param_with_count

from glob import glob
from safetensors.torch import safe_open


class Qwen3Convert(BaseModelConvert):
    """
    Convert Qwen3 model. Use https://huggingface.co/Qwen or https://modelscope.cn/models
    """

    SUPPORTED_DTYPES = {"bf16": np.uint16, "fp32": np.float32, "fp16": np.float16}

    def __init__(self):
        super().__init__()
        self.default_dtype = "bf16"

        self.hf_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            # MOE
            "mlp.gate.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        ]

        self.ft_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "attention.q_norm.weight",
            "attention.k_norm.weight",
            "attention.dense.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            # MOE
            "mlp.gate.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        ]

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
            "attention.query_key_value.weight" in key
            or "attention.dense.weight" in key
            or "mlp.gate_proj.weight" in key
            or "mlp.up_proj.weight" in key
            or "mlp.down_proj.weight" in key
            # MOE
            or "gate_proj.weight" in key
            or "up_proj.weight" in key
            or "down_proj.weight" in key
            or "mlp.gate.weight" in key
        ):
            # shared weights, only need to convert the weights of rank 0
            if i == 0:
                save_val(val, key, 0)
        elif (
            "input_layernorm.weight" in key
            or "attention.q_norm.weight" in key
            or "attention.k_norm.weight" in key
            or "post_attention_layernorm.weight" in key
        ):
            # shared weights, only need to convert the weights of rank 0
            if i == 0:
                save_val(val, key)
        else:
            print("[ERROR] cannot find key '{}'".format(key))

    def save_model_config(self, hf_config, output_dir, dtype):
        # save parameters to config file
        config = configparser.ConfigParser()
        sec_name = hf_config["model_type"]
        config[sec_name] = {}
        has_post_decoder_layernorm = True
        try:
            config[sec_name]["model_name"] = hf_config["_name_or_path"] if hf_config["_name_or_path"] else "qwen3"
            num_attention_heads = config[sec_name]["head_num"] = str(hf_config["num_attention_heads"])
            num_key_value_heads = config[sec_name]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            config[sec_name]["hidden_size"] = str(hf_config["hidden_size"])
            config[sec_name]["size_per_head"] = str(hf_config["head_dim"])
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
            config[sec_name]["pad_id"] = str(hf_config["eos_token_id"])
            config[sec_name]["weight_data_type"] = dtype
            config[sec_name]["attn_params_type"] = "GQAttnParams"
            config[sec_name]["ffn_params_type"] = "LlamaFFNParams"
            config[sec_name]["do_qk_norm"] = "1"

            if "moe" in hf_config["model_type"]:
                config[sec_name]["sparse_experts"] = str(hf_config["num_experts"])
                config[sec_name]["moe_intermediate_size"] = str(hf_config["moe_intermediate_size"])
                config[sec_name]["num_experts_per_tok"] = str(hf_config["num_experts_per_tok"])
                config[sec_name]["ffn_params_type"] = "Qwen3MOEParams"

            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))
            sys.exit(1)

        return config

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        saved_dir = output_dir

        # create directory if not exist
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        # load the model config
        model_config = AutoConfig.from_pretrained(
            input_dir,
            trust_remote_code=True,
        )
        hf_config = vars(model_config)

        config_ini = self.save_model_config(hf_config, output_dir, dtype)
        sec_name = hf_config["model_type"]
        num_attention_heads = config_ini[sec_name]["head_num"]
        num_key_value_heads = config_ini[sec_name]["kv_head_num"]

        print("Processing ...")
        state_dict = dict()
        generator, total_tensors = get_name_and_param_with_count(Path(input_dir))
        for name, tensor in tqdm(generator, total=total_tensors, desc="Loading weights"):
            state_dict[name] = tensor
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
                model_named_parameters[name.replace("self_attn.q_proj.weight", "attention.query_key_value.weight")] = qkv
                
            # for merged weights, skip
            elif "self_attn.k_proj.weight" in name or "self_attn.v_proj.weight" in name:
                continue
            elif "embed" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            elif "layernorm" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        del state_dict

        pool = multiprocessing.Pool(processes)
        with tqdm(total=len(model_named_parameters)) as pbar:
            for name, param in model_named_parameters.items():
                if name == "model.embed_tokens.weight":
                    param.to(self.torch_dtype).view(self.torch_view_dtype).numpy().tofile(os.path.join(output_dir, "model.wte.bin"))
                    if hf_config["tie_word_embeddings"] == True:
                        param.detach().to(self.torch_dtype).view(self.torch_view_dtype).cpu().numpy().tofile(
                            os.path.join(output_dir, "model.lm_head.weight.bin")
                        )
                elif name == "model.norm.weight":
                    param.to(self.torch_dtype).view(self.torch_view_dtype).numpy().tofile(
                        os.path.join(output_dir, "model.final_layernorm.weight.bin")
                    )
                elif name == "lm_head.weight":
                    param.to(self.torch_dtype).view(self.torch_view_dtype).numpy().tofile(
                        os.path.join(saved_dir, "model.lm_head.weight.bin")
                    )
                else:
                    starmap_args = []
                    for i in range(len(self.hf_model_name_pattern)):
                        if self.hf_model_name_pattern[i] in name:
                            factor = 1
                            new_name = name.replace(self.hf_model_name_pattern[i], self.ft_model_name_pattern[i])
                            starmap_args.append(
                                (
                                    0,
                                    saved_dir,
                                    factor,
                                    new_name,
                                    param.to(self.torch_dtype).view(self.torch_view_dtype).numpy(),
                                    num_attention_heads,
                                    num_key_value_heads,
                                )
                            )
                    pool.starmap_async(self.split_and_convert_process, starmap_args, callback=lambda _: pbar.update())
            pool.close()
            pool.join()

        print(f"{saved_dir} export successful!")
