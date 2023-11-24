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


class OPTConvert(BaseModelConvert):
    """
    Convert huggingface Meta OPT model. Use https://huggingface.co/facebook/opt-13b as demo.
    """

    def __init__(self):
        super().__init__()

    def split_and_convert_process(self, i, saved_dir, factor, key, val):
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
            local_dim = (int)(val.shape[-1] / 3)

            val = val.reshape(3, local_dim)
            split_vals = np.split(val, factor, axis=-1)

            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        elif "attention.query_key_value.weight" in key:
            hidden_dim = val.shape[0]
            local_dim = (int)(val.shape[-1] / 3)

            val = val.reshape(hidden_dim, 3, local_dim)

            split_vals = np.split(val, factor, axis=-1)
            for j in range(factor):
                save_val(split_vals[j], key, i * factor + j)

        else:
            print("[ERROR] cannot find key '{}'".format(key))

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        def fuse_qkv_weight(q, k, v):
            if isinstance(q, float):
                qkv = torch.tensor((q, k, v))
            else:
                qkv = torch.cat([q, k, v], dim=-1)
            return qkv

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)

        factor = 1

        # load position_embedding from rank 0
        model = AutoModelForCausalLM.from_pretrained(input_dir, device_map="auto")

        hf_config = vars(model.config)

        num_layers = hf_config["num_hidden_layers"]

        layer_names = [name for name, param in model.named_parameters()]

        # NOTE: save parameters to config files (loaded by triton backends)
        config = configparser.ConfigParser()
        config["gpt"] = {}
        has_post_decoder_layernorm = "model.decoder.final_layer_norm.bias" in layer_names
        try:
            config["gpt"]["model_name"] = "opt" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            config["gpt"]["head_num"] = str(hf_config["num_attention_heads"])
            n_embd = hf_config["hidden_size"]
            config["gpt"]["size_per_head"] = str(n_embd // hf_config["num_attention_heads"])
            config["gpt"]["inter_size"] = str(hf_config["ffn_dim"])
            config["gpt"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config["gpt"]["num_layer"] = str(hf_config["num_hidden_layers"])
            config["gpt"]["layernorm_eps"] = "1e-5"
            config["gpt"]["layernorm_type"] = "pre_layernorm" if hf_config["do_layer_norm_before"] else "post_layernorm"
            config["gpt"]["activation_type"] = "Relu"
            config["gpt"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config["gpt"]["vocab_size"] = str(hf_config["vocab_size"])
            config["gpt"]["start_id"] = str(hf_config["bos_token_id"])
            config["gpt"]["end_id"] = str(hf_config["eos_token_id"])
            config["gpt"]["weight_data_type"] = dtype
            # config['gpt']['int8'] = str(save_int8) # really useful?
            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except:
            print(f"Fail to save the config in config.ini.")

        huggingface_model_name_pattern = [
            "self_attn_layer_norm.bias",
            "self_attn_layer_norm.weight",
            "self_attn.qkv_proj.bias",
            "self_attn.qkv_proj.weight",
            "self_attn.out_proj.bias",
            "self_attn.out_proj.weight",
            "final_layer_norm.bias",
            "final_layer_norm.weight",
            "fc1.bias",
            "fc1.weight",
            "fc2.bias",
            "fc2.weight",
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

        model_named_parameters_iter = model.named_parameters()
        model_named_parameters = dict()
        for name, param in model_named_parameters_iter:
            if "embed" in name:
                model_named_parameters[name] = param
            elif "project_in" in name:
                model_named_parameters[name] = param.permute(1, 0)
            elif "project_out" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        for l in range(num_layers):
            prefix = f"model.decoder.layers.{l}.self_attn."
            q_weight = model_named_parameters[prefix + "q_proj.weight"]
            k_weight = model_named_parameters[prefix + "k_proj.weight"]
            v_weight = model_named_parameters[prefix + "v_proj.weight"]
            q_bias = model_named_parameters[prefix + "q_proj.bias"]
            k_bias = model_named_parameters[prefix + "k_proj.bias"]
            v_bias = model_named_parameters[prefix + "v_proj.bias"]
            qkv_weight = fuse_qkv_weight(q_weight, k_weight, v_weight)
            qkv_bias = fuse_qkv_weight(q_bias, k_bias, v_bias)

            model_named_parameters[prefix + "qkv_proj.weight"] = qkv_weight
            model_named_parameters[prefix + "qkv_proj.bias"] = qkv_bias

        pool = multiprocessing.Pool(processes)
        padding_offset = 2
        for name, param in model_named_parameters.items():
            if name == "model.decoder.embed_positions.weight":
                param[padding_offset:, ...].detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.wpe.bin")
                )

            elif name == "model.decoder.embed_tokens.weight":
                if "model.decoder.project_in.weight" in model_named_parameters.keys():
                    project_in = model_named_parameters["model.decoder.project_in.weight"]
                    project_out = model_named_parameters["model.decoder.project_out.weight"]
                    torch.matmul(param, project_in).detach().cpu().numpy().astype(self.dtype).tofile(
                        os.path.join(output_dir, "model.wte.bin")
                    )
                    torch.matmul(param, project_out).detach().cpu().numpy().astype(self.dtype).tofile(
                        os.path.join(output_dir, "model.lm_head.weight.bin")
                    )

                else:
                    param.detach().cpu().numpy().astype(self.dtype).tofile(os.path.join(output_dir, "model.wte.bin"))
                    param.detach().cpu().numpy().astype(self.dtype).tofile(
                        os.path.join(output_dir, "model.lm_head.weight.bin")
                    )

            elif name == "model.decoder.final_layer_norm.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
                )
            elif name == "model.decoder.final_layer_norm.bias":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.bias.bin")
                )
            elif "project_in" in name or "project_out" in name:
                continue
            else:
                starmap_args = []
                for i in range(len(huggingface_model_name_pattern)):
                    if huggingface_model_name_pattern[i] in name:
                        new_name = name.replace("model.decoder.layers.", "layers.").replace(
                            huggingface_model_name_pattern[i], ft_model_name_pattern[i]
                        )

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
