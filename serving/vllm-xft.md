# vLLM-xft
vLLM-xFT is a fork of vLLM to  integrate the xfastertransformer backend, maintaining compatibility with most of the official vLLM's features.

## Install
```bash
pip install vllm-xft
```

## Usage
***Notice: Preload libiomp5.so is required!***

### Serving(OpenAI Compatible Server)
```shell
# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tokenizer ${TOKEN_PATH} \
        --dtype bf16 \
        --kv-cache-dtype fp16 \
        --served-model-name xft \
        --port 8000 \
        --trust-remote-code 
```
- `--max-num-batched-tokens`: max batched token, default value is max(MAX_SEQ_LEN_OF_MODEL, 2048).
- `--max-num-seqs`: max seqs batch, default is 256.  

More Arguments please refer to [vllm official docs](https://docs.vllm.ai/en/latest/models/engine_args.html)  

### Query example
```shell
  curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "xft",
  "prompt": "San Francisco is a",
  "max_tokens": 16,
  "temperature": 0
  }'
```

## Distributed(Multi-rank)
Use oneCCL's `mpirun` to run the workload. The master (`rank 0`) is the same as the single-rank above, and the slaves (`rank > 0`) should use the following command:
```bash
python -m vllm.entrypoints.slave --dtype bf16 --model ${MODEL_PATH} --kv-cache-dtype fp16
```
Please keep params of slaves align with master.

### Serving(OpenAI Compatible Server)
Here is a example on 2Socket platform, 48 cores pre socket.
```bash
# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

OMP_NUM_THREADS=48 mpirun \
        -n 1 numactl --all -C 0-47 -m 0 \
          python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --tokenizer ${TOKEN_PATH} \
            --dtype bf16 \
            --kv-cache-dtype fp16 \
            --served-model-name xft \
            --port 8000 \
            --trust-remote-code \
        : -n 1 numactl --all -C 48-95 -m 1 \
          python -m vllm.entrypoints.slave \
            --dtype bf16 \
            --model ${MODEL_PATH} \
            --kv-cache-dtype fp16
```