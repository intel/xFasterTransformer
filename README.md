# xFasterTransformer

xFasterTransformer is an exceptionally optimized solution for large language models (LLM) on the X86 platform, which is similar to FasterTransformer on the GPU platform. xFasterTransformer is able to operate in distributed mode across multiple sockets and nodes to support inference on larger models. Additionally, it provides both C++ and Python APIs, spanning from high-level to low-level interfaces, making it easy to adopt and integrate.

## Table of Contents
- [xFasterTransformer](#xfastertransformer)
  - [Table of Contents](#table-of-contents)
  - [Models overview](#models-overview)
    - [Support matrix](#support-matrix)
  - [Documents](#documents)
  - [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [Using Docker](#using-docker)
    - [Built from source](#built-from-source)
      - [Prepare Environment](#prepare-environment)
        - [Manually](#manually)
        - [Docker(Recommended)](#dockerrecommended)
      - [How to build](#how-to-build)
  - [Model Preparation](#prepare-model)
  - [API usage](#api-usage)
    - [Python API(PyTorch)](#python-apipytorch)
    - [C++ API](#c-api)
  - [How to run](#how-to-run)
    - [Single rank](#single-rank)
    - [Multi ranks](#multi-ranks)
      - [Command line](#command-line)
      - [Code](#code)
        - [Python](#python)
        - [C++](#c)
  - [Web Demo](#web-demo)
  - [Benchmark](#benchmark)
  - [Support](#support)

## Models overview
Large Language Models (LLMs) develops very fast and are more widely used in many AI scenarios. xFasterTransformer is an optimized solution for LLM inference using the mainstream and popular LLM models on Xeon. xFasterTransformer fully leverages the hardware capabilities of Xeon platforms to achieve the high performance and high scalability of LLM inference both on single socket and multiple sockets/multiple nodes.

xFasterTransformer provides a series of APIs, both of C++ and Python, for end users to integrate xFasterTransformer into their own solutions or services directly. Many kinds of example codes are also provided to demonstrate the usage. Benchmark codes and scripts are provided for users to show the performance. Web demos for popular LLM models are also provided.


### Support matrix

|  Models  | Framework |          | Distribution | DataType |          |          |           |           |
| :------: | :-------: | :------: | :----------: | :------: | :------: | :------: | :-------: | :-------: |
|          |  PyTorch  |   C++    |              |   FP16   |   BF16   |   INT8   | BF16+FP16 | BF16+INT8 |
| ChatGLM  | &#10004;  | &#10004; |   &#10004;   | &#10004; | &#10004; | &#10004; | &#10004;  | &#10004;  |
| ChatGLM2 | &#10004;  | &#10004; |   &#10004;   | &#10004; | &#10004; | &#10004; | &#10004;  | &#10004;  |
|  Llama   | &#10004;  | &#10004; |   &#10004;   | &#10004; | &#10004; | &#10004; | &#10004;  | &#10004;  |
|  Llama2  | &#10004;  | &#10004; |   &#10004;   | &#10004; | &#10004; | &#10004; | &#10004;  | &#10004;  |
|   Opt    | &#10004;  | &#10004; |   &#10004;   | &#10004; | &#10004; | &#10004; | &#10004;  | &#10004;  |

## Documents
xFasterTransformer Documents and [Wiki](https://github.com/intel/xFasterTransformer/wiki) provides the following resources:
- An introduction to xFasterTransformer.
- Comprehensive API references for both high-level and low-level interfaces in C++ and PyTorch.
- Practical API usage examples for xFasterTransformer in both C++ and PyTorch.

## Installation
### From PyPI
```bash
pip install xfastertransformer
```

### Using Docker
```bash
docker pull intel/xfastertransformer:latest
```

### Built from source
#### Prepare Environment
##### Manually
- [oneCCL](https://github.com/oneapi-src/oneCCL)
  - Use provided scripts to build it from source code. 
    ```bash
    cd 3rdparty
    sh prepare_oneccl.sh
    source ./oneCCL/build/_install/env/setvars.sh
    ```
  - Install oneCCL through installing [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- [PyTorch](https://pytorch.org/get-started/locally/) v2.0+ (When using the PyTorch API, it's required, but it's not needed when using the C++ API.)
  ```bash 
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

##### Docker(Recommended)
- Pull docker image from dockerhub
  ```bash
  docker pull intel/xfastertransformer:dev-ubuntu22.04
  ```
- Build docker image from Dockerfile
  ```bash
  docker build \
  -f dockerfiles/Dockerfile \
  --build-arg "HTTP_PROXY=${http_proxy}" \
  --build-arg "HTTPS_PROXY=${https_proxy}" \
  -t intel/xfastertransformer:dev-ubuntu22.04 .
  ```
Then run the docker with the command or bash script in repo (Assume model files are in `/data/` directory):  
```bash
# A new image will be created to ensure both the user and file directories are consistent with the host if the user is not root.
bash run_dev_docker.sh

# or run docker manually by following command.
docker run -it \
    --name xfastertransformer-dev \
    --privileged \
    --shm-size=16g \
    -v "${PWD}":/root/xfastertransformer \
    -v /data/:/data/ \
    -w /root/xfastertransformer \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    intel/xfastertransformer:dev-ubuntu22.04
```
**Notice!!!**: Please enlarge `--shm-size` if  **bus error** occurred while running in the multi-ranks mode . The default docker limits the shared memory size to 64MB and our implementation uses many shared memories to achieve a  better performance.

##### How to build
- Using 'CMake'
  ```bash
  # Build xFasterTransformer
  git clone https://github.com/intel/xFasterTransformer.git xFasterTransformer
  cd xFasterTransformer
  git checkout <latest-tag>
  # Please make sure torch is installed when run python example
  mkdir build && cd build
  cmake ..
  make -j
  ```
- Using 'python setup.py'
  ```bash
  # Build xFasterTransformer library and C++ example.
  python setup.py build

  # Install xFasterTransformer into pip environment.
  python setup.py install
  ```

## [Models Preparation](tools/README.md)
xFasterTransformer supports a different model format from Huggingface, but it's compatible with FasterTransformer's format.
1. Download the huggingface format model firstly.
2. After that, convert the model into xFasterTransformer format using the script in 'tools' folder. You will see many bin files in the output directory.
```bash
    python ./tools/chatglm_convert.py -i ${HF_DATASET_DIR} -o  ${OUTPUT_DIR}

```

## API usage
For more details, please see API document and [examples](examples/README.md).
### Python API(PyTorch)
Firstly, please install the dependencies.
- Python dependencies
  ```bash
  pip install -r requirements.txt
  ```
- oneCCL  
  Install oneCCL and setup the environment. Please refer to [Prepare Environment](#prepare-environment).


xFasterTransformer's Python API is similar to transformers and also supports transformers's streamer to achieve the streaming output. In the example, we use transformers to encode input prompts to token ids. 
```Python
import xfastertransformer
from transformers import AutoTokenizer, TextStreamer
# Assume huggingface model dir is `/data/chatglm-6b-hf` and converted model dir is `/data/chatglm-6b-cpu`.
MODEL_PATH="/data/chatglm-6b-cpu"
TOKEN_PATH="/data/chatglm-6b-hf"

INPUT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."
tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH, use_fast=False, padding_side="left", trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False)

input_ids = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=False).input_ids
model = xfastertransformer.AutoModel.from_pretrained(MODEL_PATH, dtype="bf16")
generated_ids = model.generate(input_ids, max_length=200, streamer=streamer)
```

### C++ API
[SentencePiece](https://github.com/google/sentencepiece) can be used to tokenizer and detokenizer text.
```C++
#include <vector>
#include <iostream>
#include "xfastertransformer.h"
// ChatGLM token ids for prompt "Once upon a time, there existed a little girl who liked to have adventures."
std::vector<int> input(
        {3393, 955, 104, 163, 6, 173, 9166, 104, 486, 2511, 172, 7599, 103, 127, 17163, 7, 130001, 130004});

// Assume converted model dir is `/data/chatglm-6b-cpu`.
xft::AutoModel model("/data/chatglm-6b-cpu", xft::DataType::bf16);

model.config(/*max length*/ 100, /*num beams*/ 1);
model.input(/*input token ids*/ input, /*batch size*/ 1);

while (!model.isDone()) {
    std::vector<int> nextIds = model.generate();
}

std::vector<int> result = model.finalize();
for (auto id : result) {
    std::cout << id << " ";
}
std::cout << std::endl;
```

## How to run
Recommend preloading `libiomp5.so` to get a better performance. `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after building xFasterTransformer successfully.
### Single rank
Recommend using `SINGLE_INSTANCE=1` env to avoid MPI initialization.

### Multi ranks
#### Command line
Use MPI to run in the multi-ranks mode. Here is a example on local. Install oneCCL firstly, please refer to [Prepare Environment](#prepare-environment).
```bash
OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
  -n 1 numactl -N 0  -m 0 ${RUN_WORKLOAD} : \
  -n 1 numactl -N 1  -m 1 ${RUN_WORKLOAD} 
```

#### Code
For more details, please refer to examples.
##### Python
`model.rank` can get the process's rank, `model.rank == 0` is the Master.  
For Slaves, after loading the model, the only thing needs to do is `model.generate()`. The input and generation configuration will be auto synced.
```Python
model = xfastertransformer.AutoModel.from_pretrained(MODEL_PATH, dtype="bf16")

# Slave
while True:
    model.generate()
```
##### C++
`model.getRank()` can get the process's rank, `model.getRank() == 0` is the Master.  
For Slaves, any value can be input to `model.config()` and `model.input` since Master's value will be synced.
```C++
xft::AutoModel model("/data/chatglm-6b-cpu", xft::DataType::bf16);

// Slave
while (1) {
    model.config();
    std::vector<int> input_ids;
    model.input(/*input token ids*/ input_ids, /*batch size*/ 1);

    while (!model.isDone()) {
        model.generate();
    }
}
```

## [Web Demo](examples/web_demo/README.md)
A web demo based on [Gradio](https://www.gradio.app/) is provided in repo. Now support ChatGLM, ChatGLM2 and Llama2 models.
- [Perpare the model](#prepare-model).
- Install the dependencies
  ```bash
  pip install -r examples/web_demo/requirements.txt
  ```
- Run the script corresponding to the model. After the web server started, open the output URL in the browser to use the demo. Please specify the paths of model and tokenizer directory, and data type. `transformer`'s tokenizer is used to encode and decode text so `${TOKEN_PATH}` means the huggingface model directory. This demo also support multi-rank.
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after build xFasterTransformer.
SINGLE_INSTANCE=1 LD_PRELOAD=libiomp5.so python examples/web_demo/ChatGLM.py \
                                                     --dtype=bf16 \
                                                     --token_path=${TOKEN_PATH} \
                                                     --model_path=${MODEL_PATH}
```

## [Benchmark](benchmark/README.md)

Benchmark scripts are provided to  get the model inference performance quickly.
- [Prepare the model](#prepare-model).
- Enter the folder corresponding to the model, for example
  ```bash
  cd benchmark/chatglm6b/
  ```
- Run scripts `run_${MODEL}.sh`.  Please modify the model and tokenizer path in `${MODEL}.sh` before running. 
  - Shell script will automatically check the number of numa nodes. By default, at least there are 2 nodes and 48 physics cores per node (If the system is in sub-numa status, there are 12 cores for each sub-numa).
  - By default, you will get the performance of "input token=32, output token=32, Beam_width=1, FP16".
  - If more datatype and scenarios performance needed, please modify the parameters in `${MODEL}.sh`
  - If system configuration needs modification, please change run-chatglm-6b.sh.
  - If you want the custom input, please modify the `prompt_pool.json` file.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

## Support

- xFasterTransformer email: xft.maintainer@intel.com
- xFasterTransformer [wechat](https://github.com/intel/xFasterTransformer/wiki)