# Copyright (c) 2024 Intel Corporation
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
from mlserver import MLModel
from mlserver.codecs import decode_args
from transformers import AutoTokenizer
import torch
from typing import List

# import sys

# sys.path.append("../../../src")

import xfastertransformer

DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

TOKEN_PATH = "/data/llama-2-7b-chat-hf"
MODEL_PATH = "/data/llama-2-7b-chat-cpu"
# One of DTYPE_LIST
DTYPE = "fp16"
OUTPUT_LENGTH = 128

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
EOS_ID = 2


class XFTLlama2Model(MLModel):
    async def load(self) -> bool:
        self._tokenizer = AutoTokenizer.from_pretrained(
            TOKEN_PATH, use_fast=False, padding_side="left", trust_remote_code=True
        )
        self._model = xfastertransformer.AutoModel.from_pretrained(MODEL_PATH, dtype=DTYPE)

        return True

    def create_chat_input_token(self, query):
        tokens = []
        tokens.append(self._tokenizer([f"{B_INST} {query.strip()} {E_INST}"], return_tensors="pt").input_ids)
        input_tokens = torch.cat(tokens, dim=1)
        return input_tokens

    @decode_args
    async def predict(self, questions: List[str]) -> List[str]:
        input_token_ids = self.create_chat_input_token(questions[0])
        generated_ids = self._model.generate(input_token_ids, max_length=input_token_ids.shape[-1] + OUTPUT_LENGTH)
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return response
