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


class xFTModel(MLModel):
    async def load(self) -> bool:
        self.model_path = self.settings.parameters.extra["model_path"]
        self.token_path = self.settings.parameters.extra["token_path"]
        self.dtype = self.settings.parameters.extra["dtype"]
        self.output_length = self.settings.parameters.extra["output_length"]
        self.generate_config = self.settings.parameters.extra["generate_config"]
        self.stop_words_ids = None

        # Stop words ids for QWen model
        if "qwen" in self.model_path.lower():
            self.stop_words_ids = [[33975, 25], [151643]]
            # chat model should be [[151645], [151644]]

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.token_path, use_fast=False, padding_side="left", trust_remote_code=True
        )
        self._model = xfastertransformer.AutoModel.from_pretrained(self.model_path, dtype=self.dtype)

        # Llama doesn't have padding ID.
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        return True

    def create_chat_input_token(self, query):
        # modify the input prompt if necessary
        return self._tokenizer(query, return_tensors="pt", padding=True).input_ids

    @decode_args
    async def predict(self, questions: List[str]) -> List[str]:
        input_token_ids = self.create_chat_input_token(questions)
        generated_ids = self._model.generate(
            input_token_ids,
            max_length=input_token_ids.shape[-1] + self.output_length,
            stop_words_ids=self.stop_words_ids,
            **self.generate_config,
        )
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return response
