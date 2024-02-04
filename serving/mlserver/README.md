# xFasterTransformer MLServer
[MLSever](https://mlserver.readthedocs.io/en/latest/index.html) is an open source inference server for machine learning models supporting REST and gRPC interface, and adaptive batching feature to group inference requests together on the fly.

## How to run sever
If you want to run xft model on 1 CPU node, please follow [single-rank server](single-rank/README.md).  
If you want to run xft model cross CPU nodes, please follow [multi-ranks server](multi-ranks/README.md).  

## How to run client
```
python client.py --host {MLSERVER_IP} --port {MLSERVER_PORT}
```

## [Adaptive batching](https://mlserver.readthedocs.io/en/latest/user-guide/adaptive-batching.html)
You can configure adaptive batching by `model-settings.json`.
- `max_batch_size`, that is how many requests you want to group together.
  - N, where N > 1, will create batches of up to N elements.
  - 0 or 1, will disable adaptive batching.
- `max_batch_time`, that is how much time we should wait for new requests until we reach our maximum batch size.
  - T, where T > 0, will wait T seconds at most.
  - 0, will disable adaptive batching.