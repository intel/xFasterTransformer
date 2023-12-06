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
Enter the folder corresponding to the model and run `run_benchmark.sh`. Please export `model_name` and `precision` as env variables before running.

You can choose `model_name` as follows (default is `chatglm2-6b` if there's no export):
- llama-2 (-7b,-13b,-70b)
- llama (-7b,-13b,-30b,-65b)
- chatglm2-6b
- chatglm-6b
- baichuan2 (-7b,-13b)

You can choose `precision` as follows (default is `fp16` if there's no export):
- bf16
- bf16_fp16
- int8
- bf16_int8
- fp16

```bash
# Example of llama-2-7b with precision bf16.
cd benchmark
export precision=bf16
export model_name=llama-2-7b
bash run_benchmark.sh
```

- Shell script will automatically check number of numa nodes.
- By default, you will get the performance of "input token=32, output token=32, Beam_width=1".
- If more datatype and scenarios performance needed, please modify the parameters in `run_benchmark.sh`
- If system configuration needs modification, please change run_benchmark.sh.
- If you want the custom input, please modify the `prompt.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

