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
Enter the folder corresponding to the model and run `run_${MODEL}.sh`. Please modify the model and tokenizer path in `${MODEL}.sh` before running.
```bash
# ChatGLM for example.
cd benchmark/chatglm-6b
bash run_chatglm-6b.sh
```

- Shell script will automatically check number of numa nodes, default at least 2 nodes and there is 48 physics cores in each node (12core for subnuma).
- By default, you will get the performance of "input token=32, output token=32, Beam_width=1, FP16".
- If more datatype and scenarios performance needed, please modify the parameters in `${MODEL}.sh`
- If system configuration needs modification, please change run-chatglm-6b.sh.
- If you want the custom input, please modify the `prompt_pool.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

