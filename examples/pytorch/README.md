# Python (PyTorch) example
Python(PyTorch) example achieves end-to-end inference of the model with streaming output combining the transformer's tokenizer.

## Step 1: Prepare xFasterTransformer  
Please refer to [Installation](../../README.md#installation). This example supports use source code which means you don't need install xFasterTransformer into pip and just build xFasterTransformer library, and it will search library in src directory.

## Step 2: Prepare models  
Please refer to [Prepare model](../README.md#prepare-model)

## Step 3: Install the dependencies.
- Please refer to [Prepare Environment](#prepare-environment) to install oneCCL.
- Python dependencies.
    ```bash
    # requirements.txt in root directory.
    pip install -r requirements.txt
    ```

## Step 4: Run
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after build xFasterTransformer.
LD_PRELOAD=libiomp5.so python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}

# run multi-instance like
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl -N 0 -m 0 python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH} : \
  -n 1 numactl -N 1 -m 1 python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}
```
More parameter options settings:
- `-h`, `--help`            show help message and exit.
- `--token_path TOKEN_PATH` Path to tokenizer directory.
- `--model_path MODEL_PATH` Path to model directory.
- `--dtype`                 Data type, default using `fp16`, supports `{fp16,bf16,int8,bf16_fp16,bf16_int8}`.
- `--streaming`             Streaming output, Default to True.
- `--num_beams`             Num of beams, default to 1 which is greedy search.
- `--output_len`            max tokens can generate excluded input.