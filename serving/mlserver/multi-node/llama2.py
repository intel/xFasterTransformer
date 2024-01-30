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
from mlserver import MLModel, types
from mlserver.codecs import StringCodec
from mlserver.codecs import decode_args
from transformers import AutoTokenizer
import torch
from typing import List, Dict, Any
import time

import json
import grpc
import xft_pb2
import xft_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc


TOKEN_PATH = "/data/llama-2-7b-chat-hf"

XFT_IP = "localhost"
XFT_PORT = "50051"


def health_check_call(stub: health_pb2_grpc.HealthStub):
    start_time = time.time()
    request = health_pb2.HealthCheckRequest(service="xft.Runner")
    while True:
        try:
            resp = stub.Check(request)
            if resp.status == health_pb2.HealthCheckResponse.SERVING:
                return True
        except grpc._channel._InactiveRpcError as e:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            print("Health check timed out.")
            return False

        time.sleep(1)


class XFTLlama2Model(MLModel):
    async def load(self) -> bool:
        self._tokenizer = AutoTokenizer.from_pretrained(
            TOKEN_PATH, use_fast=False, padding_side="left", trust_remote_code=True
        )

        # Llama doesn't have padding ID.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self.channel = grpc.insecure_channel(f"{XFT_IP}:{XFT_PORT}")
        self.stub = xft_pb2_grpc.XFTServerStub(self.channel)
        health_stub = health_pb2_grpc.HealthStub(self.channel)
        return health_check_call(health_stub)

    def create_chat_input_token(self, query):
        return self._tokenizer(query, return_tensors="pt", padding=True).input_ids

    @decode_args
    async def predict(self, questions: List[str]) -> List[str]:
        input_token_ids = self.create_chat_input_token(questions)
        response = self.stub.predict(
            xft_pb2.QueryIds(
                Ids=input_token_ids.view(-1).tolist(),
                batch_size=input_token_ids.shape[0],
                seq_len=input_token_ids.shape[-1],
            )
        )
        response_ids = torch.Tensor(response.Ids).view(response.batch_size, response.seq_len)
        response = self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return response
    
    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )

        return inputs   

    async def predict_stream(self, payload: types.InferenceRequest):
        request = self._extract_json(payload)
        query = request["xft-llama-2-model"]["prompt"]
        input_token_ids = self.create_chat_input_token(query)

        for response in self.stub.predict_stream(
            xft_pb2.QueryIds(
                Ids=input_token_ids.view(-1).tolist(),
                batch_size=input_token_ids.shape[0],
                seq_len=input_token_ids.shape[-1],
            )
        ):
            next_token_id = response.Ids[0]
            next_token = self._tokenizer.decode(next_token_id, skip_special_tokens=True)
            if next_token != '':
                yield f"data: {next_token}\n\n"
