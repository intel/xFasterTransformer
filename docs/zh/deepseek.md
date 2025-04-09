# DeepSeek

## DeepSeek-R1蒸馏模型
蒸馏模型的使用方式与基础模型相同。例如：
- `DeepSeek-R1-Distill-Qwen-xxB`，基于 Qwen2.5，使用 `Qwen2Convert`。
- `DeepSeek-R1-Distill-Llama-xxB` 使用 `LlamaConvert`。

## DeepSeek-R1 671B满血版
注意，DeepSeek-R1 671B满血版模型只支持数据格式`fp8_e4m3`, kvcache缓存格式`bf16`。
### 软件要求
- xfastertransformer >= 2.0.0
- [oneCCL](../../README.md#command-line)
- vllm-xft >= 0.5.5.3

### 使用 Docker
注意：
- Docker 镜像仅包含 `xfastertransformer` 和 `oneCCL`，未安装 `vllm-xft`，可通过 `pip install vllm-xft` 安装。
- Docker 镜像默认设置了 `libiomp5.so` 环境变量，因此无需手动预加载 `libiomp5.so`。

```bash
docker pull intel/xfastertransformer:latest

# 使用以下命令运行 Docker（假设模型文件位于 `/data/` 目录）：
docker run -it \
    --name xfastertransformer \
    --privileged \
    --shm-size=16g \
    -v /data/:/data/ \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    intel/xfastertransformer:latest

```

### 提示
- 如果遇到 `mpirun: command not found` 错误，请参考“软件要求”部分安装 oneCCL 或使用 Docker 镜像。
- 默认情况下，基准测试脚本使用 1 个 CPU，可通过 `-s` 选项修改。
- 为了获得更好的性能，请考虑使用以下命令清除所有缓存：`sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`。
- 如果性能未达到预期，请在运行基准测试前尝试设置环境变量：`export XDNN_N64=64` 或 `export XDNN_N64=16`。

### 运行基准测试
执行基准测试脚本，使用假权重评估模型性能，无需下载超过 600GB 的权重。

#### 手动配置环境
```
    git clone https://github.com/intel/xFasterTransformer.git
    cd xFasterTransformer/benchmark

    export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')
    bash run_benchmark.sh -m deepseek-r1 -d fp8_e4m3 -kvd bf16 -bs 1 -in 32 -out 32 -s 1
```

#### 在Docker容器中
```
    cd /root/xFasterTransformer/benchmark
    bash run_benchmark.sh -m deepseek-r1 -d fp8_e4m3 -kvd bf16 -bs 1 -in 32 -out 32 -s 1

```
- `-bs`：batch size大小。
- `-in 32`：输入令牌长度，`[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`。
- `-out`：输出长度。
- `-s`：使用的 CPU 节点数量。如果只有一个启用了 SNC-3 的节点，它将用作为 subnuma 的数量。

### 模型服务（兼容 OpenAI 的服务）
#### 准备模型
- 从 HuggingFace 下载原始的 DeepSeek-R1 671B 模型。
- 将模型转换为 xFT 格式。
    ```
    python -c 'import xfastertransformer as xft; xft.DeepSeekR1Convert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
    ```
    >转换后，如果希望节省存储空间，`${HF_DATASET_DIR}` 中的 `*.safetensor` 文件不再需要。

#### 运行服务
- 单实例  
    如果希望在一个CPU numa 中运行 DeepSeek，例如 `NUMA node0 CPU(s):   0-47,96-143`
    ```bash
    # 通过以下命令预加载 libiomp5.so 或手动 LD_PRELOAD=libiomp5.so
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
    - `MODEL_PATH`：xFT 格式的模型权重。
    - `TOKEN_PATH`：与分词器相关的文件，例如 `HF_DATASET_DIR`。
    - `served-model-name`：API 中使用的模型名称。

- 分布式（多 rank）
    如果希望在多个 numa 之间运行 DeepSeek，例如
    ```
    NUMA node0 CPU(s):   0-47,96-143
    NUMA node1 CPU(s):   48-95,144-191
    ```

    ```bash
    # 通过以下命令预加载 libiomp5.so 或手动 LD_PRELOAD=libiomp5.so
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
- 查询示例
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