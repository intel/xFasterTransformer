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

## Benchmarking vLLM-xFT

### Downloading the vLLM
```bash
git clone https://github.com/Duyi-Wang/vllm.git && cd vllm/benchmarks
```

### Downloading the ShareGPT dataset
You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Benchmark offline inference throughput.
This script is used to benchmark the offline inference throughput of a specified model. It sets up the environment, defines the paths for the tokenizer, model, and dataset, and uses numactl to bind the process to appropriate CPU resources for optimized performance.
```bash
#!/bin/bash

# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# Define the paths for the tokenizer and the model
TOKEN_PATH=/data/models/Qwen2-7B-Instruct
MODEL_PATH=/data/models/Qwen2-7B-Instruct-xft
DATASET_PATH=ShareGPT_V3_unfiltered_cleaned_split.json

# Use numactl to bind to appropriate CPU resources
numactl -C 0-47 -l python benchmark_throughput.py \
        --tokenizer ${TOKEN_PATH} \          # Path to the tokenizer
        --model ${MODEL_PATH} \              # Path to the model
        --dataset ${DATASET_PATH}            # Path to the dataset
```

### Benchmark online serving throughput.
This guide explains how to benchmark the online serving throughput for a model. It includes instructions for setting up the server and running the client benchmark script.
1. On the server side, you can refer to the following code to start the test API server:
```bash
#!/bin/bash

# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# Define the paths for the tokenizer and the model
TOKEN_PATH=/data/models/Qwen2-7B-Instruct
MODEL_PATH=/data/models/Qwen2-7B-Instruct-xft

# Start the API server using numactl to bind to appropriate CPU resources
numactl -C 0-47 -l python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \              # Path to the model
        --tokenizer ${TOKEN_PATH} \          # Path to the tokenizer
        --dtype bf16 \                       # Data type for the model (bfloat16)
        --kv-cache-dtype fp16 \              # Data type for the key-value cache (float16)
        --served-model-name xft \            # Name for the served model
        --port 8000 \                        # Port number for the API server
        --trust-remote-code                  # Trust remote code execution
```

2. On the client side, you can use `python benchmark_serving.py --help` to see the required configuration parameters. Here is a reference example:

```bash
$ python benchmark_serving.py --model xft --tokenizer /data/models/Qwen2-7B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
============ Serving Benchmark Result ============
Successful requests:                     xxxx
Benchmark duration (s):                  xxxx
Total input tokens:                      xxxx
Total generated tokens:                  xxxx
Request throughput (req/s):              xxxx
Input token throughput (tok/s):          xxxx
Output token throughput (tok/s):         xxxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxxx
Median TTFT (ms):                        xxxx
P99 TTFT (ms):                           xxxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P99 TPOT (ms):                           xxx
==================================================
```