#!/bin/bash

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

