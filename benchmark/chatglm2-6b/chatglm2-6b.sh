#!/bin/bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_type=bf16_fp16
ilen=3294
olen=512
echo "${data_type} Performance, input len ${ilen}, output len ${olen}"
python "${SCRIPT_DIR}"/../benchmark.py \
    --token_path /data/chatglm2-6b \
    --model_path /data/chatglm2-6b/cpu \
    --prompt_path "${SCRIPT_DIR}"/prompt_pool.json \
    --model_name "ChatGLM2-6B" \
    --dtype ${data_type} \
    --token_in ${ilen} 	\
    --token_out ${olen} --beam_width 1 --iteration 20

# In this benchmark case, token_in only can be "9","32","64","128","256","512","1196","2050","3294"
# "32" means the token length is 32, if needs more test, add it into input_token.py

