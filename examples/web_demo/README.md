# Web Demo
a Web demo based on [Gradio](https://gradio.app) is provided.

## How to use

1. Install Gradio firstly.
```bash
pip install gradio
```

2. Install `xfastertransformer` wheel package or build from source.
3. run demo and specify params.
```bash
python web.py --dtype=fp16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}
```