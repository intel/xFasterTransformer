#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "FP16 Performance "
python "${SCRIPT_DIR}"/../benchmark.py \
    --token_path /data/llama-13b \
    --model_path /data/llama-13b/cpu \
    --prompt_path "${SCRIPT_DIR}"/prompt_pool.json \
    --model_name "Llama-13B" \
    --dtype fp16 \
    --token_in 32 	\
    --token_out 32 --beam_width 1 --iteration 100 

# In this benchmark case, token_in only can be "demo","32","64","128","256","512","1024","2016"
# "32" means the token length is 32, if needs more test, add it into input_token.py

