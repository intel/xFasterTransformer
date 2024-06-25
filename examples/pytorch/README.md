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
    ***PS: Due to the potential compatibility issues between the model file and the `transformers` version, please select the appropriate `transformers` version.***

## Step 4: Run
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# or LD_PRELOAD=libiomp5.so manually, `libiomp5.so` file will be in `3rdparty/mkl/lib` directory after build xFasterTransformer.
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# run single instance like
python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}

# run multi-rank like
OMP_NUM_THREADS=48 mpirun \
  -n 1 numactl -N 0 -m 0 python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH} : \
  -n 1 numactl -N 1 -m 1 python demo.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}
```
More parameter options settings:
- `-h`, `--help`            show help message and exit.
- `-t`, `--token_path`      Path to tokenizer directory.
- `-m`, `--model_path`      Path to model directory.
- `-d`, `--dtype`           Data type, default using `fp16`, supports `{fp16, bf16, int8, w8a8, int4, nf4, bf16_fp16, bf16_int8, bf16_w8a8,bf16_int4, bf16_nf4, w8a8_int8, w8a8_int4, w8a8_nf4}`.
- `--streaming`             Streaming output, Default to True.
- `--num_beams`             Num of beams, default to 1 which is greedy search.
- `--output_len`            max tokens can generate excluded input.
- `--padding`               Enable tokenizer padding, Default to True.
- `--chat`                  Enable chat mode for ChatGLM models, Default to False.
- `--do_sample`             Enable sampling search, Default to False.
- `--temperature`           value used to modulate next token probabilities.
- `--top_p`                 retain minimal tokens above topP threshold.
- `--top_k`                 num of highest probability tokens to keep for generation.