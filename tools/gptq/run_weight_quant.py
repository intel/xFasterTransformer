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
import os
import torch

from gptq import *

# TODO: move to tests/ut folder in xFT
if __name__ == '__main__':
    weight = torch.tensor([
        [0.123456, 1.234567, 3.456789],
        [4.567891, 5.678912, 6.789123],
        [7.891234, 8.912345, 9.123456]
    ]).float()

    llm_gptq = LLM_GPTQ(weight, 8, False)
    quantized_weight, scale, zero = llm_gptq.fasterquant()
    print("quantized weight is ")
    print(quantized_weight)
    print("scale is ")
    print(scale)
    print("zero is ")
    print(zero)
    
