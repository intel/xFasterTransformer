# Single rank MLServer

## requirement
```
pip install mlserver
```

## 1. Configure model setting
Edit model file path and generation params in the `model-settings.json`
```json
"model_path": "/data/llama-2-7b-chat-cpu",
"token_path": "/data/llama-2-7b-chat-hf",
"dtype": "fp16",
"output_length": 512,
"generate_config": {
    "num_beams": 1,
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0
}
```

## 2. Start MLServer
Please choose the appropriate values for `OMP_NUM_THREADS` based on the specific hardware environment.
```bash
cd mlserver/single-rank
LD_PRELOAD=libiomp5.so OMP_NUM_THREADS=48 numactl -C 0-48 -m 0 mlserver start .
```