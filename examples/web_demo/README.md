# Web Demo
A web demo based on [Gradio](https://www.gradio.app/) is provided in repo. 

Support models list:
- ChatGLM
- ChatGLM2
- Llama2-chat

## Step 1: Prepare xFasterTransformer  
Please refer to [Installation](../../README.md#installation). This example supports use source code which means you don't need install xFasterTransformer into pip and just build xFasterTransformer library, and it will search library in src directory.

## Step 2: Prepare models  
Please refer to [Prepare model](../README.md#prepare-model)

## Step 3: Install the dependencies.
- Please refer to [Prepare Environment](#prepare-environment) to install oneCCL.
- Python dependencies.
    ```bash
    # requirements.txt in `examples/web_demo/`.
    pip install -r requirements.txt
    ```

## Step 4: Run the script corresponding to the model. 
After the web server started, open the output URL in the browser to use the demo. Please specify the paths of model and tokenizer directory, and data type. `transformer`'s tokenizer is used to encode and decode text so `${TOKEN_PATH}` means the huggingface model directory.
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after build xFasterTransformer.
LD_PRELOAD=libiomp5.so python examples/web_demo/ChatGLM.py \
                                    --dtype=bf16 \
                                    --token_path=${TOKEN_PATH} \
                                    --model_path=${MODEL_PATH}

# run multi-rank like
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl -N 0 -m 0 python examples/web_demo/ChatGLM.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}: \
  -n 1 numactl -N 1 -m 1 python examples/web_demo/ChatGLM.py --dtype=bf16 --token_path=${TOKEN_PATH} --model_path=${MODEL_PATH}: 
```
