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
Enter the folder corresponding to the model and run `run_benchmark.sh`. Please modify the `model_name` in `run_benchmark.sh` before running.
```bash
# Benchmark example.
cd benchmark
bash run_benchmark.sh
```

- Shell script will automatically check number of numa nodes.
- By default, you will get the performance of "input token=32, output token=32, Beam_width=1, FP16".
- If more datatype and scenarios performance needed, please modify the parameters in `run_benchmark.sh`
- If system configuration needs modification, please change run_benchmark.sh.
- If you want the custom input, please modify the `prompt.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

