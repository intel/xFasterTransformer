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
from torch import nn

from transformers import AutoModelForCausalLM

from .convert import BaseModelConvert


class BaichuanConvert(BaseModelConvert):
    """
    Convert huggingface Baichuan model. Use https://huggingface.co/baichuan-inc
    """

    def __init__(self):
        super().__init__()

    def split_and_convert_process(self, i, saved_dir, factor, key, val):
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

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load the model
        model = AutoModelForCausalLM.from_pretrained(input_dir, device_map="auto", trust_remote_code=True)

        hf_config = vars(model.config)

        # save parameters to config file
        config = configparser.ConfigParser()
        config["baichuan"] = {}
        has_post_decoder_layernorm = True
        try:
            config["baichuan"]["model_name"] = (
                "baichuan" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            )
            config["baichuan"]["head_num"] = str(hf_config["num_attention_heads"])
            hidden_size = hf_config["hidden_size"]
            config["baichuan"]["size_per_head"] = str(hidden_size // hf_config["num_attention_heads"])
            config["baichuan"]["inter_size"] = str(hf_config["intermediate_size"])
            config["baichuan"]["max_pos_seq_len"] = str(hf_config.get("max_position_embeddings", 0))
            config["baichuan"]["model_max_length"] = str(
                hf_config.get("model_max_length", config["baichuan"]["max_pos_seq_len"])
            )
            config["baichuan"]["num_layer"] = str(hf_config["num_hidden_layers"])
            config["baichuan"]["rms_norm_eps"] = "1e-6"
            config["baichuan"]["layernorm_type"] = "pre_layernorm"
            config["baichuan"]["activation_type"] = "silu"
            config["baichuan"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config["baichuan"]["vocab_size"] = str(hf_config["vocab_size"])
            config["baichuan"]["start_id"] = str(hf_config["bos_token_id"])
            config["baichuan"]["end_id"] = str(hf_config["eos_token_id"])
            config["baichuan"]["weight_data_type"] = dtype
            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))

        hf_model_name_pattern = [
            "input_layernorm.weight",
            "self_attn.W_pack.weight",
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
            if "embed" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = nn.functional.normalize(param)
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        pool = multiprocessing.Pool(processes)
        for name, param in model_named_parameters.items():
            if name == "model.embed_tokens.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(os.path.join(output_dir, "model.wte.bin"))
            elif name == "model.norm.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
                )
            # elif name == 'model.final_layernorm.bias':
            #     param.detach().cpu().numpy().astype(self.dtype).tofile(
            #         os.path.join(output_dir, "model.final_layernorm.bias.bin"))
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
                            )
                        )
                pool.starmap_async(self.split_and_convert_process, starmap_args)
        pool.close()
        pool.join()
