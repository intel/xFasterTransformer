# Multi ranks MLServer

## requirement
```bash
pip install mlserver grpcio grpcio-health-checking grpcio-tools
```

## 1. Start xft grpc server
Follow [xft grpc server](../../grpc_launcher/README.md) to start a grpc xft server with multi-ranks using scripts in `grpc_launcher` with `mpirun`.

## 2. Configure model setting
Edit params in `model-settings.json`.
```json
"token_path": "/data/llama-2-7b-chat-hf",
"xft_grpc_server_ip": "localhost",
"xft_grpc_server_port": "50051"
```

## 3. Start MLServer
```bash
cd mlserver/multi-ranks
mlserver start .
```