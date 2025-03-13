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

from transformers import AutoModelForCausalLM

from .convert import BaseModelConvert


class TelechatConvert(BaseModelConvert):
    """
    Convert huggingface Telechat model.
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

        elif "mlp.gate_proj" in key or "mlp.up_proj" in key or "mlp.down_proj" in key:
            split_vals = np.split(val, factor, axis=0)
            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        elif "attention.query_key_value" in key:
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

        elif "attention.dense" in key:
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
        model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )

        hf_config = vars(model.config)

        # save parameters to config file
        config = configparser.ConfigParser()
        # sec_name = hf_config["model_type"]
        sec_name = "telechat"
        config[sec_name] = {}
        has_post_decoder_layernorm = True
        try:
            config[sec_name]["model_name"] = "llama" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            num_attention_heads = config[sec_name]["head_num"] = str(hf_config["n_head"])
            num_key_value_heads = config[sec_name]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            hidden_size = hf_config["hidden_size"]
            config[sec_name]["hidden_size"] = str(hidden_size)
            config[sec_name]["size_per_head"] = str(hidden_size // hf_config["n_head"])
            config[sec_name]["inter_size"] = str(hf_config["ffn_hidden_size"])
            config[sec_name]["max_pos_seq_len"] = str(hf_config["seq_length"])
            config[sec_name]["num_layer"] = str(hf_config["n_layer"])
            config[sec_name]["layernorm_eps"] = str(hf_config.get("layer_norm_epsilon", 1e-6))
            config[sec_name]["layernorm_type"] = "pre_layernorm"
            config[sec_name]["activation_type"] = "silu"
            config[sec_name]["rope_theta"] = str(hf_config.get("rope_theta", 10000))
            rope_scaling = hf_config.get("rope_scaling", None)
            if rope_scaling:
                config[sec_name]["scaling_factor"] = str(rope_scaling.get("factor", 1.0))
                config[sec_name]["rope_type"] = str(rope_scaling.get("type", "null"))
            else:
                config[sec_name]["scaling_factor"] = str(1.0)
                config[sec_name]["rope_type"] = str("null")
            config[sec_name]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config[sec_name]["vocab_size"] = str(hf_config["vocab_size"])
            config[sec_name]["start_id"] = str(hf_config["bos_token_id"])
            config[sec_name]["end_id"] = str(hf_config["eos_token_id"])
            pad_token_id = hf_config.get("pad_token_id")
            if pad_token_id is not None:
                config[sec_name]["pad_id"] = str(pad_token_id)
            config[sec_name]["weight_data_type"] = dtype
            config[sec_name]["attn_params_type"] = "GQAttnParams"
            config[sec_name]["ffn_params_type"] = "LlamaFFNParams"
            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))
        hf_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "self_attention.dense.weight",
            "self_attention.dense.bias",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.bias",
        ]

        ft_model_name_pattern = [
            "input_layernorm.weight",
            "attention.query_key_value.weight",
            "attention.dense.weight",
            "attention.dense.bias",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.bias",
        ]

        state_dict = model.state_dict()
        model_named_parameters = dict()
        n_heads = int(num_attention_heads)
        head_dim = int(hidden_size) // n_heads
        for name, param in state_dict.items():
            name = name.replace("transformer.h", "model.layers")
            print(f"name = {name}")
            # merge QKV
            if "self_attention.query.weight" in name:
                kv_name = name.replace("query", "key_value").replace("model.layers", "transformer.h")
                kv_param = state_dict[kv_name].permute(1, 0).reshape(-1, n_heads, head_dim * 2)
                k_param, v_param = torch.split(kv_param, head_dim, dim=-1)
                k_param = k_param.reshape(-1, k_param.shape[0])
                v_param = v_param.reshape(-1, v_param.shape[0])
                qkv = torch.cat((param.permute(1, 0), k_param, v_param), dim=1)
                model_named_parameters[
                    name.replace("self_attention.query.weight", "attention.query_key_value.weight")
                ] = qkv
            # for merged weights, skip
            elif "self_attention.key_value.weight" in name:
                continue
            elif "word_embeddings" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            elif "layernorm" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        pool = multiprocessing.Pool(processes)
        for name, param in model_named_parameters.items():
            if name == "transformer.word_embeddings.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(os.path.join(output_dir, "model.wte.bin"))
            elif name == "transformer.ln_f.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
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
                        new_name = name.replace(hf_model_name_pattern[i], ft_model_name_pattern[i])
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

    def split_and_convert_quantized_model(self, input_dir, output_dir, dtype, processes, from_quantized_model):
        raise NotImplementedError("Quantized model conversion for telechat model is not implemented yet.")
