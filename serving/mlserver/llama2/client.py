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

DEFAULT_QUERY = "Hello! How are you today?"
B_INST, E_INST = "[INST]", "[/INST]"

while True:
    input_prompt = input("[Please enter the query]: ")
    if input_prompt == "":
        input_prompt = DEFAULT_QUERY
        print("[Use default query]:" + input_prompt)

    inference_request = {
        "inputs": [StringCodec.encode_input(name="questions", payload=[input_prompt], use_bytes=False).dict()]
    }

    r = requests.post("http://127.0.0.1:8080/v2/models/xft-llama-2-model/infer", json=inference_request)

    print("[Response]:" + r.json()["outputs"][0]["data"][0].split(E_INST, 1)[1].lstrip())
