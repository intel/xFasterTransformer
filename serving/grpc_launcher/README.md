# GRPC Lanucher
Run a grpc xfastertransformer server, this server accepts token ids as input.

## Install requirements
```bash
pip install grpcio grpcio-health-checking grpcio-tools
```

## How to run server
### single-rank
```bash
LD_PRELOAD=libiomp5.so python server.py --model_path ${MODEL_PATH}
```

### multi-ranks
Use MPI to run in the multi-ranks mode, please install oneCCL firstly.  
The reference for running commands is as follows. Please choose the appropriate number of ranks and values for `OMP_NUM_THREADS` based on the specific hardware environment
```bash
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl --all -C 0-47 -m 0 python server.py --model_path ${MODEL_PATH} : \
  -n 1 numactl --all -C 48-95 -m 1 python server.py --model_path ${MODEL_PATH}
```

### More parameter options settings
- `-h`, `--help`            show help message and exit.
- `--port`                  serve port, default 50051.
- `-m`, `--model_path`      Path to model directory.
- `-d`, `--dtype`           Data type, default using `fp16`, supports `{fp16, bf16, int8, w8a8, int4, nf4, bf16_fp16, bf16_int8, bf16_w8a8,bf16_int4, bf16_nf4, w8a8_int8, w8a8_int4, w8a8_nf4}`.
- `--num_beams`             Num of beams, default to 1 which is greedy search.
- `--output_len`            max tokens can generate excluded input.
- `--do_sample`             Enable sampling search, Default to False.
- `--temperature`           value used to modulate next token probabilities.
- `--top_p`                 retain minimal tokens above topP threshold.
- `--top_k`                 num of highest probability tokens to keep for generation.
- `--rep_penalty`           param for repetition penalty. 1.0 means no penalty

## How to run client
port default using 50051. Notice that streaming generation only supports batch szie 1.
```bash
python client.py --token_path {TOKEN_PATH} --port {GRPC_PORT}

# for streaming generation
python client_stream.py --token_path {TOKEN_PATH} --port {GRPC_PORT}
```

## Message format
- Request
```
message QueryIds {
  repeated int32 Ids = 1;
  int32 batch_size = 2;
  int32 seq_len = 3;
}
```
- Response
```
message ResponseIds {
  repeated int32 Ids = 1;
  int32 batch_size = 2;
  int32 seq_len = 3;
}
```