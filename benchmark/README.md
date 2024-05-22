# Benchmark

Benchmark scripts is provided to quickly get the model inference performance.

## Step 1: Prepare xFasterTransformer  
Please refer to [Installation](../../README.md#installation). This example supports using source code which means you don't need install xFasterTransformer into pip and just build xFasterTransformer library, and it will search library in src directory.

## Step 2: Prepare models  
Please refer to [Prepare model](../README.md#prepare-model)

## Step 3: Install the dependencies.
- Please refer to [Prepare Environment](#prepare-environment) to install oneCCL.
- Python dependencies.
    ```bash
    # requirements.txt in root directory.
    pip install -r requirements.txt
    ```

## Step 4: Run scripts
Enter the folder corresponding to the model and run `run_benchmark.sh -m <model_name>`.

Please choose `<model_name>` as follows:
- llama-2 (-7b,-13b,-70b)
- llama (-7b,-13b,-30b,-65b)
- chatglm2-6b
- chatglm3-6b
- chatglm-6b
- baichuan2 (-7b,-13b)

Please choose `-d` or `--dtype` as follows:
- bf16 (default)
- bf16_fp16
- int8
- bf16_int8
- fp16
- bf16_int4
- int4
- bf16_nf4
- nf4
- bf16_w8a8
- w8a8
- w8a8_int8
- w8a8_int4
- w8a8_nf4

Please choose `-s` or `--sockets` as follows:
- 1 (default, benchmarking on single socket)
- 2 (benchmarking on 2 sockets)

Specify data type of kvcache using `-kdv` or `--kv_cache_dtype` from below list:
- fp16 (default)
- int8

Specify batch size using `-mp` or `--model_path`. (If not been specified, will use fake model config)
Specify batch size using `-tp` or `--token_path`. (If not been specified, will use fake tokenizer config)
Specify batch size using `-bs` or `--batch_size`. (default 1)
Specify input tokens using `-in` or `--input_tokens`. (default 32)
Specify output tokens using `-out` or `--output_tokens`. (default 32)
Specify beam width using `-b` or `--beam_width`. (default 1)
Specify inference iteration using `-i` or `--iter`. (default 10)


```bash
# Example of llama-2-7b with precision bf16, batch size 1, 1024 input tokens and 128 output tokens on single socket.
cd benchmark
# setup mpirun env
source ../3rdparty/oneccl/build/_install/env/setvars.sh
bash run_benchmark.sh -m llama-2-7b -d bf16 -s 1 -bs 1 -in 1024 -out 128 -i 10
```

- Shell script will automatically check number of numa nodes.

- If system configuration needs modification, please change run_benchmark.sh.
- If you want the custom input, please modify the `prompt.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.


## Step 4: Run distributed scripts

- Ensure identical physical hardware, and the network is on the same subnet.
- [Optional] Use NFS to store code and ensure a consistent environment.
- Enable passwordless SSH between machines.
- Maintain the IP <-> hosts mapping in each device `/etc/hosts` file.

```bash
bash -x run_benchmark_dist.sh
```