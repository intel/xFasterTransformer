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

from transformers import AutoModel

from .convert import BaseModelConvert


class ChatGLM2Convert(BaseModelConvert):
    """
    Convert huggingface ChatGLM2 model. Use https://huggingface.co/THUDM/chatglm2-6b
    """

    def __init__(self):
        super().__init__()
        self.model_type = "chatglm2"

    def split_and_convert_process(
        self,
        i,
        saved_dir,
        factor,
        key,
        val,
        num_attention_heads,
        multi_query_group_num,
        kv_channels,
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

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load the model
        model = AutoModel.from_pretrained(input_dir, trust_remote_code=True)

        hf_config = vars(model.config)
        print("hf_config= {}".format(hf_config))

        layer_names = [name for name, param in model.named_parameters()]

        # save parameters to config file
        config = configparser.ConfigParser()
        config[self.model_type] = {}
        has_post_decoder_layernorm = "model.decoder.final_layer_norm.bias" in layer_names
        try:
            config[self.model_type]["model_name"] = (
                self.model_type if hf_config["_name_or_path"] == "" else hf_config["_name_or_path"]
            )
            num_attention_heads = config[self.model_type]["head_num"] = str(hf_config["num_attention_heads"])
            n_embd = hf_config["hidden_size"]
            config[self.model_type]["size_per_head"] = str(n_embd // hf_config["num_attention_heads"])
            config[self.model_type]["inter_size"] = str(hf_config["ffn_hidden_size"])
            config[self.model_type]["max_pos_seq_len"] = str(hf_config["seq_length"])
            config[self.model_type]["num_layer"] = str(hf_config["num_layers"])
            config[self.model_type]["layernorm_eps"] = str(hf_config["layernorm_epsilon"])  # "1e-5"
            config[self.model_type]["layernorm_type"] = "pre_layernorm"
            config[self.model_type]["activation_type"] = "swiglu"
            config[self.model_type]["has_post_decoder_layernorm"] = (
                "1" if str(hf_config["post_layer_norm"]) == "True" else "0"
            )
            config[self.model_type]["vocab_size"] = str(hf_config["padded_vocab_size"])
            config[self.model_type]["start_id"] = str(hf_config["bos_token_id"])
            config[self.model_type]["end_id"] = str(hf_config["eos_token_id"])
            config[self.model_type]["weight_data_type"] = dtype

            kv_channels = config[self.model_type]["kv_channels"] = str(hf_config["kv_channels"])
            config[self.model_type]["rmsnorm"] = "1" if str(hf_config["rmsnorm"]) == "True" else "0"
            config[self.model_type]["apply_residual_connection_post_layernorm"] = (
                "1" if str(hf_config["apply_residual_connection_post_layernorm"]) == "True" else "0"
            )
            config[self.model_type]["multi_query_attention"] = (
                "1" if str(hf_config["multi_query_attention"]) == "True" else "0"
            )
            multi_query_group_num = config[self.model_type]["kv_head_num"] = str(hf_config["multi_query_group_num"])
            config[self.model_type]["pad_id"] = str(hf_config["pad_token_id"])

            with open(os.path.join(output_dir, "config.ini"), "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print("Fail to save the config in config.ini.", str(e))

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

        pool = multiprocessing.Pool(processes)
        for name, param in model_named_parameters.items():
            if name == "transformer.embedding.word_embeddings.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(os.path.join(output_dir, "model.wte.bin"))
            elif name == "transformer.encoder.final_layernorm.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.final_layernorm.weight.bin")
                )
            elif name == "transformer.output_layer.weight":
                param.detach().cpu().numpy().astype(self.dtype).tofile(
                    os.path.join(output_dir, "model.lm_head.weight.bin")
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
                                output_dir,
                                factor,
                                new_name,
                                param.detach().cpu().numpy().astype(self.dtype),
                                num_attention_heads,
                                multi_query_group_num,
                                kv_channels,
                            )
                        )
                pool.starmap_async(self.split_and_convert_process, starmap_args)
        pool.close()
        pool.join()
