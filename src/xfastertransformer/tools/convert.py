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
import numpy as np
import os
from pathlib import Path
import json
import traceback
import transformers

from typing import Any, Callable, ContextManager, Iterator, Sequence, TypeVar, cast


def check_transformers_version_compatibility(token_path):
    config_path = os.path.join(token_path, "config.json")
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)

        transformers_version = config_data.get("transformers_version")
    except Exception as e:
        pass
    else:
        if transformers_version:
            if transformers.__version__ != transformers_version:
                print(
                    f"[Warning] The version of `transformers` in model configuration is {transformers_version}, and version installed is {transformers.__version__}. "
                    + "This model convert error may be caused by transformers version compatibility. "
                    + f"You can downgrade or reinstall transformers by `pip install transformers=={transformers_version} --force-reinstall` and try again."
                )


def get_name_and_param(model_dir: Path):
    all_files = os.listdir(model_dir)
    safetensors_files = [f for f in all_files if f.endswith(".safetensors")]
    num_parts = len(safetensors_files)
    file_list = (
        [f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1)]
        if num_parts > 1
        else safetensors_files
    )
    print(f"Found {num_parts} model parts: {file_list}")
    for part_name in file_list:
        ctx: ContextManager[Any]
        from safetensors import safe_open

        ctx = cast(
            ContextManager[Any],
            safe_open(Path(model_dir) / part_name, framework="pt", device="cpu"),
        )
        with ctx as model_part:
            for name in model_part.keys():
                yield name, model_part.get_tensor(name)


class BaseModelConvert:
    def __init__(self):
        self.dtype = np.float32

    def __call__(self, input_dir, output_dir=None, dtype: str = "fp16", processes=8, from_quantized_model=None):
        self.convert(input_dir, output_dir, dtype, processes, from_quantized_model)

    def get_weight_data_type(self, dtype: str):
        if dtype == "fp32":
            return np.float32
        elif dtype == "fp16":
            return np.float16
        else:
            raise Exception(f"{self.__class__.__name__} don't support convert weight to {dtype}.")

    # from_quantized_model: Convert from HuggingFace quantized int8/int4 model to xFT int8/int4 model.
    #     - "gptq" : Convert from AutoGPTQ quantized model.
    def convert(self, input_dir, output_dir=None, dtype: str = "fp16", processes=8, from_quantized_model=None):
        self.dtype = self.get_weight_data_type(dtype)
        if output_dir is None:
            input_dir = input_dir.rstrip(os.path.sep)
            output_dir = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + "-xft")
        try:
            if from_quantized_model is None:
                self.split_and_convert(input_dir, output_dir, dtype, processes)
            else:
                self.split_and_convert_quantized_model(input_dir, output_dir, dtype, processes, from_quantized_model)
        except Exception as e:
            traceback.print_exc()
            check_transformers_version_compatibility(input_dir)

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        pass

    def split_and_convert_quantized_model(self, input_dir, output_dir, dtype, processes, from_quantized_model):
        raise Exception(f"{self.__class__.__name__} does not support converting from quantized model.")
