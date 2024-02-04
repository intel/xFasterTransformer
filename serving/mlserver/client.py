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
from mlserver.codecs import StringCodec
import requests
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--host", help="serve ip, default 127.0.0.1.", type=str, default="127.0.0.1")
parser.add_argument("--port", help="serve port, default 8080.", type=int, default=8080)

args = parser.parse_args()

DEFAULT_QUERY = "Hello! How are you today?"

input_prompt = DEFAULT_QUERY

inference_request = {
    "inputs": [StringCodec.encode_input(name="questions", payload=[input_prompt], use_bytes=False).dict()]
}

r = requests.post(f"http://{args.host}:{args.port}/v2/models/xft-model/infer", json=inference_request)

print(f"[Input_prompt]:{input_prompt}\n" + "-" * 40 + "\n[Response]:" + r.json()["outputs"][0]["data"][0])
