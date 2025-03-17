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
from glob import glob
from safetensors.torch import safe_open, load_file

from transformers import AutoModelForCausalLM, AutoConfig

from .convert import BaseModelConvert


class DeepSeekV2Convert(BaseModelConvert):
    """
    Convert DeepSeek V2/V3 model.
    """

    SUPPORTED_DTYPES = {"bf16": np.uint16, "fp8_e4m3": np.uint8}

    def __init__(self):
        super().__init__()
        self.default_dtype = "fp8_e4m3"
        self.hf_model_name_pattern = [
            "input_layernorm.weight",
            # MLA
            ## for DeepSeekV2lite, using q_a_proj.weight instead of q_proj.weight
            "self_attn.q_proj.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            # MLP
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.gate.weight",
            "mlp.experts",
            "mlp.shared_experts",
        ]

        self.ft_model_name_pattern = [
            "input_layernorm.weight",
            # MLA
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "attention.dense.weight",
            "post_attention_layernorm.weight",
            # MLP
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.gate.weight",
            "mlp.experts",
            "mlp.shared_experts",
        ]

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
            or "self_attn.q_proj.weight" in key
            or "self_attn.q_a_proj.weight" in key
            or "self_attn.q_a_layernorm.weight" in key
            or "self_attn.q_b_proj.weight" in key
            or "self_attn.kv_a_proj_with_mqa.weight" in key
            or "self_attn.kv_a_layernorm.weight" in key
            or "self_attn.kv_b_proj.weight" in key
            or "attention.dense.weight" in key
            or "post_attention_layernorm.weight" in key
            or "mlp.gate_proj.weight" in key
            or "mlp.up_proj.weight" in key
            or "mlp.down_proj.weight" in key
            or "mlp.gate.weight" in key
            or "mlp.experts" in key
            or "mlp.shared_experts" in key
        ):
            # shared weights, only need to convert the weights of rank 0
            if i == 0:
                save_val(val, key)
        else:
            print("[ERROR] cannot find key '{}'".format(key))

    def save_model_config(self, hf_config, output_dir, dtype):
        # save parameters to config file
        config = configparser.ConfigParser()
        sec_name = "deepseek_moe"  # hf_config["model_type"]
        config[sec_name] = {}
        has_post_decoder_layernorm = True
        try:
            config[sec_name]["model_name"] = (
                hf_config["_name_or_path"] if hf_config["_name_or_path"] else "deepseek_moe"
            )
            config[sec_name]["head_num"] = str(hf_config["num_attention_heads"])
            num_attention_heads = config[sec_name]["head_num"]
            config[sec_name]["kv_head_num"] = str(hf_config.get("num_key_value_heads", num_attention_heads))
            num_key_value_heads = config[sec_name]["kv_head_num"]

            hidden_size = hf_config["hidden_size"]
            config[sec_name]["hidden_size"] = str(hidden_size)
            config[sec_name]["size_per_head"] = str(hf_config["v_head_dim"])
            config[sec_name]["inter_size"] = str(hf_config["intermediate_size"])
            config[sec_name]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config[sec_name]["num_layer"] = str(hf_config["num_hidden_layers"])
            config[sec_name]["layernorm_eps"] = str(hf_config.get("rms_norm_eps", 1e-6))
            config[sec_name]["layernorm_type"] = "pre_layernorm"
            config[sec_name]["activation_type"] = str(hf_config["hidden_act"])
            config[sec_name]["rope_theta"] = str(hf_config.get("rope_theta", 10000))
            rope_scaling = hf_config.get("rope_scaling", None)
            config[sec_name]["rope_scaling_factor"] = str(rope_scaling.get("factor", 1.0))
            config[sec_name]["rope_scaling_type"] = str(rope_scaling.get("type", "null"))
            config[sec_name]["rope_scaling_original_max_position_embeddings"] = str(
                rope_scaling.get("original_max_position_embeddings", 4096)
            )
            config[sec_name]["rope_scaling_beta_fast"] = str(rope_scaling.get("beta_fast", 32))
            config[sec_name]["rope_scaling_beta_slow"] = str(rope_scaling.get("beta_slow", 1))
            config[sec_name]["rope_scaling_mscale"] = str(rope_scaling.get("mscale", 1.0))
            config[sec_name]["rope_scaling_mscale_all_dim"] = str(rope_scaling.get("mscale_all_dim", 1.0))
            config[sec_name]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config[sec_name]["vocab_size"] = str(hf_config["vocab_size"])
            config[sec_name]["start_id"] = str(hf_config["bos_token_id"])
            config[sec_name]["end_id"] = str(hf_config["eos_token_id"])
            config[sec_name]["pad_id"] = str(hf_config["eos_token_id"])
            config[sec_name]["weight_data_type"] = dtype
            config[sec_name]["attn_params_type"] = "MLAttnParams"
            config[sec_name]["ffn_params_type"] = "DeepSeekFFNParams"
            # for MLA
            config[sec_name]["q_lora_rank"] = str(hf_config.get("q_lora_rank", 0))
            config[sec_name]["kv_lora_rank"] = str(hf_config["kv_lora_rank"])
            config[sec_name]["qk_rope_head_dim"] = str(hf_config["qk_rope_head_dim"])
            config[sec_name]["v_head_dim"] = str(hf_config["v_head_dim"])
            config[sec_name]["qk_nope_head_dim"] = str(hf_config["qk_nope_head_dim"])
            # for MOE
            config[sec_name]["moe_intermediate_size"] = str(hf_config["moe_intermediate_size"])
            config[sec_name]["first_k_dense_replace"] = str(hf_config["first_k_dense_replace"])
            ## n_shared_experts
            config[sec_name]["dense_experts"] = str(hf_config["n_shared_experts"])
            ## n_routed_experts
            config[sec_name]["sparse_experts"] = str(hf_config["n_routed_experts"])
            config[sec_name]["routed_scaling_factor"] = str(hf_config["routed_scaling_factor"])
            config[sec_name]["topk_method"] = str(hf_config["topk_method"])
            config[sec_name]["n_group"] = str(hf_config["n_group"])
            config[sec_name]["topk_group"] = str(hf_config["topk_group"])
            config[sec_name]["num_experts_per_tok"] = str(hf_config["num_experts_per_tok"])
            config[sec_name]["norm_topk_prob"] = str(hf_config["norm_topk_prob"])
            config[sec_name]["scoring_func"] = str(hf_config["scoring_func"])

            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))

        return config

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # HF can't load fp8_e4m3 model now
        if dtype == "fp8_e4m3":
            return self.split_and_convert_fp8_e4m3(input_dir, output_dir, dtype, processes)

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )

        hf_config = vars(model.config)
        config_ini = self.save_model_config(hf_config, output_dir, dtype)
        num_attention_heads = config_ini["deepseek_moe"]["head_num"]
        num_key_value_heads = config_ini["deepseek_moe"]["kv_head_num"]

        state_dict = model.state_dict()
        model_named_parameters = dict()
        for name, param in state_dict.items():
            # print(f"name = {name}")
            if "embed" in name:
                # model.embed_tokens.weight
                model_named_parameters[name] = param
            elif "norm" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            elif "bias" in name:
                model_named_parameters[name] = param
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param

        pool = multiprocessing.Pool(processes)
        with tqdm(total=len(model_named_parameters)) as pbar:
            for name, param in model_named_parameters.items():
                if name == "model.embed_tokens.weight":
                    param.detach().cpu().view(torch.uint16).numpy().tofile(os.path.join(output_dir, "model.wte.bin"))
                    pbar.update()
                elif name == "model.norm.weight":
                    param.detach().cpu().view(torch.uint16).numpy().tofile(
                        os.path.join(output_dir, "model.final_layernorm.weight.bin")
                    )
                    pbar.update()
                elif name == "lm_head.weight":
                    param.detach().cpu().view(torch.uint16).numpy().tofile(
                        os.path.join(output_dir, "model.lm_head.weight.bin")
                    )
                    pbar.update()
                elif "mlp.gate.e_score_correction_bias" in name:
                    param.detach().cpu().view(torch.uint16).numpy().tofile(os.path.join(output_dir, name + ".bin"))
                    pbar.update()
                else:
                    starmap_args = []
                    for i in range(len(self.hf_model_name_pattern)):
                        if self.hf_model_name_pattern[i] in name:
                            factor = 1
                            new_name = name.replace(self.hf_model_name_pattern[i], self.ft_model_name_pattern[i])
                            try:
                                val = param.detach().cpu().view(torch.uint16).numpy()
                            except Exception as e:
                                print(f"Fail to convert params {name}.", str(e))
                            starmap_args.append(
                                (
                                    0,
                                    output_dir,
                                    factor,
                                    new_name,
                                    val,
                                    num_attention_heads,
                                    num_key_value_heads,
                                )
                            )
                    pool.starmap_async(self.split_and_convert_process, starmap_args, callback=lambda _: pbar.update())

            pool.close()
            pool.join()

    def split_and_convert_fp8_e4m3(self, input_dir, output_dir, dtype, processes):
        assert dtype == "fp8_e4m3"

        # e_score_correction_bias: torch.float32    -> torch.bfloat16
        # scale_inv:torch.float32                   -> torch.float32
        # mlp.gate.weight: torch.bfloat16           -> torch.bfloat16
        # norm: torch.bfloat16                      -> torch.bfloat16
        # embed: torch.bfloat16                     -> torch.bfloat16
        # lm_head: torch.bfloat16                   -> torch.bfloat16
        # others: torch.float8_e4m3fn               -> torch.float8_e4m3fn

        # load the model config
        model_config = AutoConfig.from_pretrained(
            input_dir,
            trust_remote_code=True,
        )

        hf_config = vars(model_config)
        assert "quantization_config" in hf_config
        assert hf_config["quantization_config"]["activation_scheme"] == "dynamic"
        assert hf_config["quantization_config"]["fmt"] == "e4m3"
        assert hf_config["quantization_config"]["quant_method"] == "fp8"
        assert hf_config["quantization_config"]["weight_block_size"] == [128, 128]

        config_ini = self.save_model_config(hf_config, output_dir, dtype)
        num_attention_heads = config_ini["deepseek_moe"]["head_num"]
        num_key_value_heads = config_ini["deepseek_moe"]["kv_head_num"]

        pool = multiprocessing.Pool(processes)
        weights_num = 0
        for file_path in glob(os.path.join(input_dir, "*.safetensors")):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                num = len(f.keys())
                weights_num += num

        with tqdm(total=weights_num) as pbar:
            for file_path in glob(os.path.join(input_dir, "*.safetensors")):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        weight = f.get_tensor(name)
                        if name == "model.embed_tokens.weight":
                            weight.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "model.wte.bin"))
                            pbar.update()
                        elif name == "model.norm.weight":
                            weight.view(torch.uint16).numpy().tofile(
                                os.path.join(output_dir, "model.final_layernorm.weight.bin")
                            )
                            pbar.update()
                        elif name == "lm_head.weight":
                            weight.view(torch.uint16).numpy().tofile(
                                os.path.join(output_dir, "model.lm_head.weight.bin")
                            )
                            pbar.update()
                        elif "mlp.gate.e_score_correction_bias" in name:
                            weight.to(torch.bfloat16).view(torch.uint16).numpy().tofile(
                                os.path.join(output_dir, name + ".bin")
                            )
                            pbar.update()
                        else:
                            if len(weight.shape) == 2:
                                weight = weight.permute(1, 0)
                            starmap_args = []
                            for i in range(len(self.hf_model_name_pattern)):
                                if self.hf_model_name_pattern[i] in name:
                                    factor = 1
                                    new_name = name.replace(
                                        self.hf_model_name_pattern[i], self.ft_model_name_pattern[i]
                                    )
                                    try:
                                        if name.endswith("_scale_inv"):
                                            # fp32
                                            val = weight.cpu().numpy()
                                        elif "norm" in name or "mlp.gate.weight" in name:
                                            # bf16
                                            val = weight.cpu().view(torch.uint16).numpy()
                                        else:
                                            # fp8_e4m3
                                            val = weight.cpu().view(torch.uint8).numpy()
                                    except Exception as e:
                                        print(f"Fail to convert params {name}.", str(e))
                                    starmap_args.append(
                                        (
                                            0,
                                            output_dir,
                                            factor,
                                            new_name,
                                            val,
                                            num_attention_heads,
                                            num_key_value_heads,
                                        )
                                    )
                                    break
                            pool.starmap_async(
                                self.split_and_convert_process, starmap_args, callback=lambda _: pbar.update()
                            )
            pool.close()
            pool.join()

    def split_and_convert_quantized_model(self, input_dir, output_dir, dtype, processes, from_quantized_model):
        raise NotImplementedError("Quantized model conversion for DeepSeekV2 model is not implemented yet.")
