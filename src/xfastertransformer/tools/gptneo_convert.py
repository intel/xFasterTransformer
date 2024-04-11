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

        elif "mlp.dense_h_to_4h" in key or "mlp.dense_4h_to_h" in key:
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
        config["gpt_neo"] = {}
        has_post_decoder_layernorm = True
        try:
            config["gpt_neo"]["model_name"] = "gptneo" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            num_attention_heads = config["gpt_neo"]["head_num"] = str(hf_config["num_heads"])
            num_key_value_heads = config["gpt_neo"]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            hidden_size = hf_config["hidden_size"]
            inter_size = hf_config.get("intermediate_size", None)
            inter_size = hidden_size*4 if inter_size == None else inter_size

            config["gpt_neo"]["size_per_head"] = str(hidden_size // hf_config["num_heads"])
            config["gpt_neo"]["inter_size"] = str(inter_size)
            config["gpt_neo"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config["gpt_neo"]["num_layer"] = str(hf_config["num_layers"])
            config["gpt_neo"]["layernorm_eps"] = str(hf_config.get("layer_norm_epsilon", 1e-5))
            config["gpt_neo"]["layernorm_type"] = "pre_layernorm"
            config["gpt_neo"]["activation_type"] = "gelu"
            config["gpt_neo"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config["gpt_neo"]["vocab_size"] = str(hf_config["vocab_size"])
            config["gpt_neo"]["start_id"] = str(hf_config["bos_token_id"])
            config["gpt_neo"]["end_id"] = str(hf_config["eos_token_id"])
            config["gpt_neo"]["weight_data_type"] = dtype
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

    def split_and_convert_quantized_model(self, input_dir, output_dir, dtype, processes, from_quantized_model):
        """
        Convert from AutoGPTQ quantized int8/int4 model to xFT int8/int4 model.
        """

        if from_quantized_model != "gptq":
            print(f"[ERROR] Input model must be AutoGPTQ quantized model. from_quantized_model must be 'gptq'.")
            return

        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load AutoGPTQ quantized model
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            input_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        hf_config = vars(model.config)
        quantize_config = vars(model.quantize_config)

        # save parameters to config file
        config = configparser.ConfigParser()
        config["gpt_neo"] = {}
        has_post_decoder_layernorm = True
        try:
            config["gpt_neo"]["model_name"] = "gptneo" if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            num_attention_heads = config["gpt_neo"]["head_num"] = str(hf_config["num_heads"])
            num_key_value_heads = config["gpt_neo"]["kv_head_num"] = str(
                hf_config.get("num_key_value_heads", num_attention_heads)
            )

            hidden_size = hf_config["hidden_size"]
            inter_size = hf_config.get("intermediate_size", None)
            inter_size = hidden_size*4 if inter_size == None else inter_size

            config["gpt_neo"]["size_per_head"] = str(hidden_size // hf_config["num_heads"])
            config["gpt_neo"]["inter_size"] = str(inter_size)
            config["gpt_neo"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
            config["gpt_neo"]["num_layer"] = str(hf_config["num_layers"])
            config["gpt_neo"]["layernorm_eps"] = str(hf_config.get("layer_norm_epsilon", 1e-5))
            config["gpt_neo"]["layernorm_type"] = "pre_layernorm"
            config["gpt_neo"]["activation_type"] = "gelu"
            config["gpt_neo"]["has_post_decoder_layernorm"] = "1" if has_post_decoder_layernorm else "0"
            config["gpt_neo"]["vocab_size"] = str(hf_config["vocab_size"])
            config["gpt_neo"]["start_id"] = str(hf_config["bos_token_id"])
            config["gpt_neo"]["end_id"] = str(hf_config["eos_token_id"])
            config["gpt_neo"]["weight_data_type"] = dtype

            self.wbits = quantize_config["bits"]
            assert self.wbits == 8 or self.wbits == 4, "Only 4/8bits quantization is supported"
            config["gpt_neo"]["quant_qweight_data_type"] = 'int8' if self.wbits == 8 else 'uint4'
            config["gpt_neo"]["quant_scales_data_type"] = 'fp32'
            config["gpt_neo"]["quant_zeros_data_type"] = 'fp32'
            assert quantize_config["group_size"] == -1, "Only column wise quantization is supported."
            config["gpt_neo"]["quant_groupsize"] = str(quantize_config["group_size"])
            #config["gpt_neo"]["quant_scheme"] = "sym" if quantize_config["sym"] == True else "asym"

            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))

        hf_model_name_pattern = [
            "ln_1.weight",
            "ln_1.bias",
            "attn.attention.query_key_value.qweight",
            "attn.attention.query_key_value.qzeros",
            "attn.attention.query_key_value.scales",
            "attn.attention.out_proj.qweight",
            "attn.attention.out_proj.qzeros",
            "attn.attention.out_proj.scales",
            "attn.attention.out_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.qweight",
            "mlp.c_fc.qzeros",
            "mlp.c_fc.scales",
            "mlp.c_fc.bias",
            "mlp.c_proj.qweight",
            "mlp.c_proj.qzeros",
            "mlp.c_proj.scales",
            "mlp.c_proj.bias",
        ]

        ft_model_name_pattern = [
            "input_layernorm.weight",
            "input_layernorm.bias",
            "attention.query_key_value.qweight",
            "attention.query_key_value.zeros",
            "attention.query_key_value.scales",
            "attention.dense.qweight",
            "attention.dense.zeros",
            "attention.dense.scales",
            "attention.dense.bias",
            "post_attention_layernorm.weight",
            "post_attention_layernorm.bias",
            "mlp.dense_h_to_4h.qweight",
            "mlp.dense_h_to_4h.zeros",
            "mlp.dense_h_to_4h.scales",
            "mlp.dense_h_to_4h.bias",
            "mlp.dense_4h_to_h.qweight",
            "mlp.dense_4h_to_h.zeros",
            "mlp.dense_4h_to_h.scales",
            "mlp.dense_4h_to_h.bias",
        ]

        state_dict = model.state_dict()
        model_named_parameters = dict()

        # merge QKV
        new_state_dict = dict()
        for name, param in state_dict.items():
            if "attn.attention.q_proj" in name:
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                qkv = torch.cat((param, state_dict[k_name], state_dict[v_name]), dim=-1)
                new_state_dict[name.replace("q_proj", "query_key_value")] = qkv
            elif "attn.attention.k_proj" in name or "attn.attention.v_proj" in name:
                continue
            else:
                new_state_dict[name] = param
        state_dict = new_state_dict

        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[6:]
            wf = torch.tensor(list(range(0, 32, self.wbits)), dtype=torch.int32).unsqueeze(0)

            print(name)

            if "embed" in name:
                model_named_parameters[name] = param
            elif "lm_head" in name:
                model_named_parameters[name] = param
            elif "scales" in name:
                # scales is fp16 in AutoQPTQ, convert to fp32 for xFT.
                model_named_parameters[name] = param.float()
            elif "qzeros" in name:
                # get qzeros
                qzeros = param
                qzeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // self.wbits),
                        wf.unsqueeze(0)).to(torch.int16 if self.wbits == 8 else torch.int8)
                qzeros = qzeros + 1
                qzeros = torch.bitwise_and(qzeros, (2 ** self.wbits) - 1)

                # qzeoros is uint8/uint4 in AutoQPTQ, zeros is fp32 for xFT.
                # for int8, zeros = - scales * (qzeros - 128)
                # for uint4, zeros = - scales * qzeros
                if self.wbits == 8:
                    qzeros = qzeros - 128 # uint8 to int8
                qzeros = torch.flatten(qzeros).float()
                scales = state_dict["model." + name.replace("qzeros", "scales")].float()
                zeros = - scales * qzeros
                model_named_parameters[name] = zeros
            elif "qweight" in name:
                # get qweight
                qweight = param
                qweight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // self.wbits, -1),
                        wf.unsqueeze(-1)).to(torch.int16 if self.wbits == 8 else torch.int8)
                qweight = torch.bitwise_and(qweight, (2 ** self.wbits) - 1)
                qweight = qweight.reshape(-1, qweight.shape[2])

                # qweight is uint8/uint4, not transposed in AutoGPTQ
                # qweight is int8/uint4x2 in xFT
                if self.wbits == 8:
                    # uint8 to int8
                    qweight = qweight - 128
                else:
                    # pack uint4 to uint4x2
                    qweight = qweight.view(torch.int16)
                    qweight = qweight.bitwise_or(qweight.bitwise_right_shift(4))
                    qweight = torch.bitwise_and(qweight, 255)
                model_named_parameters[name] = qweight.to(torch.int8)
            else:
                model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param
            print(name)

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
                dtype = self.dtype
                if "qweight" in name:
                    dtype = np.int8
                if "qzero" in name or "scales" in name:
                    dtype = np.float32
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
                                param.detach().cpu().numpy().astype(dtype),
                                num_attention_heads,
                                num_key_value_heads,
                            )
                        )
                pool.starmap_async(self.split_and_convert_process, starmap_args)
        pool.close()
        pool.join()
