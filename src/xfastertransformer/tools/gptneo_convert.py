# Copyright (c) 2023 Intel Corporation
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

from transformers import GPTNeoForCausalLM

from .convert import BaseModelConvert


class GPTNeoConvert(BaseModelConvert):
    """
    Convert huggingface GPT-neo model.
    """

    def __init__(self):
        super().__init__()

    def split_and_convert_process(self, i, output_dir, factor, key, val, num_attention_heads, num_key_value_heads):
        def save_val(val, key, tp_num=None):
            if key.startswith("model."):
                path = os.path.join(output_dir, key)
            else:
                path = os.path.join(output_dir, "model." + key)

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

        elif "mlp.dense_h_to_4h.bias" in key or "mlp.dense_h_to_4h.weight" in key or "mlp.dense_4h_to_h.weight" in key:
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

        elif "attention.dense.weight" in key:
            split_vals = np.split(val, factor, axis=0)
            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        else:
            print("[ERROR] cannot find key '{}'".format(key))

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load the model
        model = GPTNeoForCausalLM.from_pretrained(
            input_dir,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        hf_config = vars(model.config)

        # save parameters to config file
        config = configparser.ConfigParser()
        config["gpt"] = {}
        has_post_decoder_layernorm = True
        try:
            config["gpt"]["model_name"] = "gptneo" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            num_attention_heads = config["gpt"]["head_num"] = str(hf_config["num_heads"])
            num_key_value_heads = config["gpt"]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            hidden_size = hf_config["hidden_size"]
            inter_size = hf_config.get("intermediate_size", None) 
            inter_size = hidden_size*4 if inter_size == None else inter_size

            config["gpt"]["size_per_head"] = str(hidden_size // hf_config["num_heads"])
            config["gpt"]["inter_size"] = str(inter_size)
            config["gpt"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config["gpt"]["num_layer"] = str(hf_config["num_layers"])
            config["gpt"]["layernorm_eps"] = str(hf_config.get("layer_norm_epsilon", 1e-5))
            config["gpt"]["layernorm_type"] = "pre_layernorm"
            config["gpt"]["activation_type"] = "gelu"
            config["gpt"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config["gpt"]["vocab_size"] = str(hf_config["vocab_size"])
            config["gpt"]["start_id"] = str(hf_config["bos_token_id"])
            config["gpt"]["end_id"] = str(hf_config["eos_token_id"])
            config["gpt"]["weight_data_type"] = dtype
            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))

        hf_model_name_pattern = [
            "ln_1.weight",
            "ln_1.bias",
            "attn.attention.query_key_value.weight",
            "attn.attention.out_proj.weight",
            "attn.attention.out_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ]

        ft_model_name_pattern = [
            "input_layernorm.weight",
            "input_layernorm.bias",
            "attention.query_key_value.weight",
            "attention.dense.weight",
            "attention.dense.bias",
            "post_attention_layernorm.weight",
            "post_attention_layernorm.bias",
            "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias",
            "mlp.dense_4h_to_h.weight",
            "mlp.dense_4h_to_h.bias",
        ]

        state_dict = model.state_dict()
        model_named_parameters = dict()
        for name, param in state_dict.items():
            print(name)
            # merge QKV
            if "attn.attention.q_proj.weight" in name:
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                qkv = torch.cat(
                    (param.permute(1, 0), state_dict[k_name].permute(1, 0), state_dict[v_name].permute(1, 0)), dim=1
                )
                model_named_parameters[
                    name.replace("attn.attention.q_proj.weight", "attn.attention.query_key_value.weight")
                ] = qkv
            # for merged weights, skip
            elif "attn.attention.k_proj.weight" in name or "attn.attention.v_proj.weight" in name:
                continue
            elif "embed" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        pool = multiprocessing.Pool(processes)
        for name, param in model_named_parameters.items():
            if name == "transformer.wte.weight":
                param.detach().cpu().numpy().astype(self.dtype).transpose().tofile(os.path.join(output_dir, "model.wte.bin"))
            elif name == "transformer.wpe.weight":
                param.detach().cpu().numpy().astype(self.dtype).transpose().tofile(os.path.join(output_dir, "model.wpe.bin"))
            elif name == "transformer.ln_f.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
                )
            elif name == "transformer.ln_f.bias":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.bias.bin")
                )
            elif name == "lm_head.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.lm_head.weight.bin")
                )
            else:
                starmap_args = []
                for i in range(len(hf_model_name_pattern)):
                    if hf_model_name_pattern[i] in name:
                        factor = 1
                        #new_name = name.replace(hf_model_name_pattern[i], ft_model_name_pattern[i])
                        new_name = name.replace("transformer.h.", "model.layers.").replace(
                            hf_model_name_pattern[i], ft_model_name_pattern[i]
                        )
                        starmap_args.append(
                            (
                                0,
                                output_dir,
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
