# C++ example
C++ example support automatic identification model and tokenizer which is implemented by [SentencePiece](https://github.com/google/sentencepiece), excluding Opt model which tokenizer is a hard code. 

## Step 1: Build binary file  
Please refer to [Build from source](../README.md#built-from-source) to build C++ example binary which is built with xFasterTransformer library and under `build` directory named `example`.

## Step 2: Prepare models  
Please refer to [Prepare model](../README.md#prepare-model)

## Step 3: Run binary  
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after build xFasterTransformer.
LD_PRELOAD=libiomp5.so ./example -m ${MODEL_PATH} -t ${TOKEN_PATH}

# run multi-instance like
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl -N 0 -m 0 ./example -m ${MODEL_PATH} -t ${TOKEN_PATH} : \
  -n 1 numactl -N 1 -m 1 ./example -m ${MODEL_PATH} -t ${TOKEN_PATH} 
```
More parameter options settings:
-   `-?`, `-h`, `--help`    Help information
-   `-m`, `--model`         directory path of xft format model.
-   `-t`, `--token`         path of tokenizer file(name like tokenizer.model), invalid for Opt model.
-   `-i`, `--input`         input prompt, invalid for Opt model. Default use `Once upon a time, there existed a little girl who liked to have adventures.`                                                                           
-   `-d`, `--dtype`         data type, default `fp16`, should be one of `["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"]`
-   `-l`, `--input_len`     input token size. Input token ids will ben expand to this size if it greater than  input prompt's size.
-   `-n`, `--num_beams`     number of beam size, default 1.
-   `-b`, `--batch_size`    batch size, default 1. If greater than 1, input prompt will be duplicated this times. 
-   `--output_len`    max tokens can generate excluded input, default 100.
-   `--loop`          number of loop, default 10.
-   `--no_stream`     disable streaming output.