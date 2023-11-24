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
import numpy as np
import os


class BaseModelConvert:
    def __init__(self):
        self.dtype = np.float32

    def __call__(self, input_dir, output_dir=None, dtype: str = "fp16", processes=8):
        self.convert(input_dir, output_dir, dtype, processes)

    def get_weight_data_type(self, dtype: str):
        if dtype == "fp32":
            return np.float32
        elif dtype == "fp16":
            return np.float16
        else:
            raise Exception(f"{self.__class__.__name__} don't support convert weight to {dtype}.")

    def convert(self, input_dir, output_dir=None, dtype: str = "fp16", processes=8):
        self.dtype = self.get_weight_data_type(dtype)
        if output_dir is None:
            input_dir = input_dir.rstrip(os.path.sep)
            output_dir = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + "-xft")

        self.split_and_convert(input_dir, output_dir, dtype, processes)

    def split_and_convert(self, input_dir, output_dir, dtype, processes):
        pass
