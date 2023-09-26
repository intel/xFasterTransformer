# Setp 1: Preparation xfastertransformer
## Build from source
```
    # cd into root directory
    # please make sure torch is installed
    mkdir build
    cd build && cmake ..
    make
```

## Install release wheel package
Download wheel package from [Release Page](https://github.com/intel-sandbox/ai.llm.llm-opt/releases).

# Setp 2: Run
```
pip install -r src/xfastertransformer/requirements.txt

cd examples/pytorch
python demo.py --dtype=fp16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}
```

# Notice
Llama's token need 3 files : tokenizer_config.json, tokenizer.model, special_tokens_map.json

Please modify the "special_tokens_map_file" in tokenizer_config.json