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

echo "FP16 Performance "
python "${SCRIPT_DIR}"/../benchmark.py \
    --token_path /data/Baichuan2-7B-Chat \
    --model_path /data/Baichuan2-7B-Chat/cpu \
    --prompt_path "${SCRIPT_DIR}"/prompt_pool.json \
    --model_name "Baichuan2-7B" \
    --dtype fp16 \
    --token_in 32 	\
    --token_out 32 --beam_width 1 --batch_size 2 --iteration 100 --padding=False

# In this benchmark case, token_in only can be "demo","32","64","128","256","512","1024","2016"
# "32" means the token length is 32, if needs more test, add it into input_token.py

