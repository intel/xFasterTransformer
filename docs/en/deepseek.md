# DeepSeek

## DeepSeek-R1-Distill Models
The distilled models can be used in the same way as the base model. For instance:
- `DeepSeek-R1-Distill-Qwen-xxB`, based on Qwen2.5, utilizes `Qwen2Convert`.
- `DeepSeek-R1-Distill-Llama-xxB` employs `LlamaConvert`.

## DeepSeek-R1 671B
Notice: DeepSeek-R1 671B only supports dtype `fp8_e4m3` and kvcache dtype `bf16`.
### Requirements
- xfastertransformer >= 2.0.0
- [oneCCL](../../README.md#command-line)
- vllm-xft >= 0.5.5.3

### Running Benchmarks
Execute the benchmark scripts to evaluate the model's performance using fake weight without downloading 600GB+ weight.
```
    git clone https://github.com/intel/xFasterTransformer.git
    cd xFasterTransformer/benchmark

    export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')
    bash run_benchmark.sh -m deepseek-r1 -d fp8_e4m3 -kvd bf16 -bs 1 -in 32 -out 32 -s 1
```
- `-bs`: batch size.
- `-in 32`: input token length, `[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`.
- `-out`: output length.
- `-s`: number of used CPU nodes. If you only have 1 node with SNC-3 enabled, it will be used as number of subnumas.

### Serving the Model(OpenAI Compatible Server)
#### Preparing the Model
- Download original DeepSeek-R1 671B model from HuggingFace.
- Convert model into xFT format. 
    ```
    python -c 'import xfastertransformer as xft; xft.DeepSeekR1Convert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
    ```
    >After conversion, the `*.safetensor` files in `${HF_DATASET_DIR}` is no longer needed if you want to save storage space.

#### Run Serving
- Single instance  
    If you want to run DeepSeek in one CPU numa, like `NUMA node0 CPU(s):   0-47,96-143`
    ```bash
    # Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
    export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

    numactl 0-47 -l python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --tokenizer ${TOKEN_PATH} \
            --dtype fp8_e4m3 \
            --kv-cache-dtype bf16 \
            --served-model-name xft \
            --port 8000 \
            --trust-remote-code 
    ```
    - `MODEL_PATH`: The xFT format model weights.
    - `TOKEN_PATH`: The tokenizer related files, like `HF_DATASET_DIR`.
    - `served-model-name`: The model name used in the API.

- Distributed(Multi-rank)
    If you want to run DeepSeek cross numas, like 
    ```
    NUMA node0 CPU(s):   0-47,96-143
    NUMA node1 CPU(s):   48-95,144-191
    ```

    ```bash
    # Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
    export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

    OMP_NUM_THREADS=48 mpirun \
            -n 1 numactl --all -C 0-47 \
            python -m vllm.entrypoints.openai.api_server \
                --model ${MODEL_PATH} \
                --tokenizer ${TOKEN_PATH} \
                --dtype fp8_e4m3 \
                --kv-cache-dtype bf16 \
                --served-model-name xft \
                --port 8000 \
                --trust-remote-code \
            : -n 1 numactl --all -C 48-95 \
            python -m vllm.entrypoints.slave \
                --dtype fp8_e4m3 \
                --model ${MODEL_PATH} \
                --kv-cache-dtype bf16
    ```
- Query example
    ```shell
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "xft",
            "messages": [{"role": "user", "content": "你好呀！请问你是谁？"}],
            "max_tokens": 256,
            "temperature": 0.6,
            "top_p": 0.95
        }'
    ```