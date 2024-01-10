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
import requests

PORT=8096

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
EOS_ID = 2


class XFTLlama2Model(MLModel):
    async def load(self) -> bool:
        return True

    def create_chat_input_token(self, query):
        # 构造llama2-chat输入
        query = [f"{B_INST} {q.strip()} {E_INST}" for q in query]
        return query

    @decode_args
    async def predict(self, questions: List[str]) -> List[str]:
        query = self.create_chat_input_token(questions)
        print(query)
        data = {"query": query}
        response = requests.post(f"http://127.0.0.1:8096/xft/predict/", json=data)
        print(response.json())
        return response.json()["response"]

