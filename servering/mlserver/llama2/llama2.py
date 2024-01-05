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

        # Llama doesn't have padding ID.
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        return True

    def create_chat_input_token(self, query):
        # 构造llama2-chat输入
        query = [f"{B_INST} {q.strip()} {E_INST}" for q in query]
        return self._tokenizer(query, return_tensors="pt", padding=True).input_ids

    @decode_args
    async def predict(self, questions: List[str]) -> List[str]:
        input_token_ids = self.create_chat_input_token(questions)
        generated_ids = self._model.generate(input_token_ids, max_length=input_token_ids.shape[-1] + OUTPUT_LENGTH)
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return response

    async def predict_stream(self, questions: List[str]) -> List[str]:
        input_token_ids = self.create_chat_input_token(questions)

        # config()函数配置generation 配置，max_length,do_sample等等
        self._model.config(input_token_ids.shape[-1] + OUTPUT_LENGTH)
        # input()输入input_prompt的ids, torch.tensor, batch_size的形状信息从tensor.shape读取。目前多batch推理默认padding。
        self._model.input(input_token_ids)

        response_ids = input_token_ids

        # 当所有batch都推理结束的时候，is_done()返回True
        # 多batch情况下，如果有sample提前结束，会填充pad_token_id
        while not self._model.is_done():
            # forward()每次调用生成一次token, 返回[batch_size, 1]
            next_token_id = self.model.forward()
            response_ids = torch.cat((response_ids, next_token_id), dim=1)

            yield self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)
