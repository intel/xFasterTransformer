# Multi rank MLServer

## requirement
```
pip install grpcio grpcio-health-checking grpcio-tools
```

## 1. Start xft server
Start a grpc xft server with multi-rank using scripts in `grpc_launcher` with `mpirun`.
The generation config is set in xft server. 
```
cd ../../grpc_launcher

# run multi-rank like
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl -N 0 -m 0 python server.py --dtype=bf16 --model_path=${MODEL_PATH} : \
  -n 1 numactl -N 1 -m 1 python server.py --dtype=bf16 --model_path=${MODEL_PATH}
```
More parameter options settings:
- `-h`, `--help`            show help message and exit.
- `-m`, `--model_path`      Path to model directory.
- `-d`, `--dtype`           Data type, default using `fp16`, supports `{fp16, bf16, int8, w8a8, int4, nf4, bf16_fp16, bf16_int8, bf16_w8a8,bf16_int4, bf16_nf4, w8a8_int8, w8a8_int4, w8a8_nf4}`.
- `--num_beams`             Num of beams, default to 1 which is greedy search.
- `--output_len`            max tokens can generate excluded input.
- `--padding`               Enable tokenizer padding, Default to True.
- `--do_sample`             Enable sampling search, Default to False.
- `--temperature`           value used to modulate next token probabilities.
- `--top_p`                 retain minimal tokens above topP threshold.
- `--top_k`                 num of highest probability tokens to keep for generation.

## 2. Start MLServer
```
cd mlserver/multi-node
mlserver start .
```

## 3. Run Client Demo
```
python client.py
```