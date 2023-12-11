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
Enter the folder corresponding to the model and run `run_benchmark.sh -m <model_name> -d <dtype> -s <sockets>`.

Please choose `<model_name>` as follows:
- llama-2 (-7b,-13b,-70b)
- llama (-7b,-13b,-30b,-65b)
- chatglm2-6b
- chatglm3-6b
- chatglm-6b
- baichuan2 (-7b,-13b)

Please choose `<dtype>` as follows:
- bf16
- bf16_fp16
- int8
- bf16_int8
- fp16

```bash
# Example of llama-2-7b with precision bf16 on single socket.
cd benchmark
# setup mpirun env
source ../3rdparty/oneccl/build/_install/env/setvars.sh
bash run_benchmark.sh -m llama-2-7b -d bf16 -s 1
```

- Shell script will automatically check number of numa nodes.
- By default, you will get the performance of "input token=32, output token=32, Beam_width=1".
- If more scenarios performance needed, please modify the parameters in `run_benchmark.sh`
- If system configuration needs modification, please change run_benchmark.sh.
- If you want the custom input, please modify the `prompt.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

