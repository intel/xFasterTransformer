# xFasterTransformer

xFasterTransformer is an exceptionally optimized solution for large language models (LLM) on the X86 platform, which is similar to FasterTransformer on the GPU platform. xFasterTransformer is able to operate in distributed mode across multiple sockets and nodes to support inference on larger models. Additionally, it provides both C++ and Python APIs, spanning from high-level to low-level interfaces, making it easy to adopt and integrate.

## Table of Contents
- [xFasterTransformer](#xfastertransformer)
  - [Table of Contents](#table-of-contents)
  - [Models overview](#models-overview)
    - [Model support matrix](#model-support-matrix)
    - [DataType support list](#datatype-support-list)
  - [Documents](#documents)
  - [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [Using Docker](#using-docker)
    - [Built from source](#built-from-source)
      - [Prepare Environment](#prepare-environment)
        - [Manually](#manually)
        - [How to build](#how-to-build)
  - [Models Preparation](#models-preparation)
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
  - [Serving](#serving)
  - [Benchmark](#benchmark)
  - [Support](#support)
  - [Q\&A](#qa)

## Models overview
Large Language Models (LLMs) develops very fast and are more widely used in many AI scenarios. xFasterTransformer is an optimized solution for LLM inference using the mainstream and popular LLM models on Xeon. xFasterTransformer fully leverages the hardware capabilities of Xeon platforms to achieve the high performance and high scalability of LLM inference both on single socket and multiple sockets/multiple nodes.

xFasterTransformer provides a series of APIs, both of C++ and Python, for end users to integrate xFasterTransformer into their own solutions or services directly. Many kinds of example codes are also provided to demonstrate the usage. Benchmark codes and scripts are provided for users to show the performance. Web demos for popular LLM models are also provided.


### Model support matrix

|       Models       | Framework |          | Distribution |
| :----------------: | :-------: | :------: | :----------: |
|                    |  PyTorch  |   C++    |              |
|      ChatGLM       | &#10004;  | &#10004; |   &#10004;   |
|      ChatGLM2      | &#10004;  | &#10004; |   &#10004;   |
|      ChatGLM3      | &#10004;  | &#10004; |   &#10004;   |
|       Llama        | &#10004;  | &#10004; |   &#10004;   |
|       Llama2       | &#10004;  | &#10004; |   &#10004;   |
|      Baichuan      | &#10004;  | &#10004; |   &#10004;   |
|        QWen        | &#10004;  | &#10004; |   &#10004;   |
| SecLLM(YaRN-Llama) | &#10004;  | &#10004; |   &#10004;   |
|        Opt         | &#10004;  | &#10004; |   &#10004;   |

### DataType support list

- FP16
- BF16
- INT8
- W8A8
- INT4
- NF4
- BF16_FP16
- BF16_INT8
- BF16_W8A8
- BF16_INT4
- BF16_NF4
- W8A8_INT8
- W8A8_int4
- W8A8_NF4

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
Run the docker with the command (Assume model files are in `/data/` directory):  
```bash
docker run -it \
    --name xfastertransformer \
    --privileged \
    --shm-size=16g \
    -v /data/:/data/ \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    intel/xfastertransformer:latest
```
**Notice!!!**: Please enlarge `--shm-size` if  **bus error** occurred while running in the multi-ranks mode . The default docker limits the shared memory size to 64MB and our implementation uses many shared memories to achieve a  better performance.

### Built from source
#### Prepare Environment
##### Manually
- [PyTorch](https://pytorch.org/get-started/locally/) v2.0 (When using the PyTorch API, it's required, but it's not needed when using the C++ API.)
  ```bash 
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

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
- Using `python setup.py`
  ```bash
  # Build xFasterTransformer library and C++ example.
  python setup.py build

  # Install xFasterTransformer into pip environment.
  # Notice: Run `python setup.py build` before installation!
  python setup.py install
  ```

## [Models Preparation](tools/README.md)
xFasterTransformer supports a different model format from Huggingface, but it's compatible with FasterTransformer's format.
1. Download the huggingface format model firstly.
2. After that, convert the model into xFasterTransformer format by using model convert module in xfastertransformer. If output directory is not provided, converted model will be placed into `${HF_DATASET_DIR}-xft`.
    ```
    python -c 'import xfastertransformer as xft; xft.LlamaConvert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
    ```
    ***PS: Due to the potential compatibility issues between the model file and the `transformers` version, please select the appropriate `transformers` version.***
    
    Supported model convert list:
    - LlamaConvert
    - ChatGLMConvert
    - ChatGLM2Convert
    - ChatGLM3Convert
    - OPTConvert
    - BaichuanConvert
    - QwenConvert

## API usage
For more details, please see API document and [examples](examples/README.md).
### Python API(PyTorch)
Firstly, please install the dependencies.
- Python dependencies
  ```bash
  pip install -r requirements.txt
  ```
  ***PS: Due to the potential compatibility issues between the model file and the `transformers` version, please select the appropriate `transformers` version.***
- oneCCL (For multi ranks)  
  Install oneCCL and setup the environment. Please refer to [Prepare Environment](#prepare-environment).


xFasterTransformer's Python API is similar to transformers and also supports transformers's streamer to achieve the streaming output. In the example, we use transformers to encode input prompts to token ids. 
```Python
import xfastertransformer
from transformers import AutoTokenizer, TextStreamer
# Assume huggingface model dir is `/data/chatglm-6b-hf` and converted model dir is `/data/chatglm-6b-xft`.
MODEL_PATH="/data/chatglm-6b-xft"
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

// Assume converted model dir is `/data/chatglm-6b-xft`.
xft::AutoModel model("/data/chatglm-6b-xft", xft::DataType::bf16);

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
FasterTransformer will automatically check the MPI environment, or you can use the `SINGLE_INSTANCE=1` environment variable to forcefully deactivate MPI.  

### Multi ranks
#### Command line
Use MPI to run in the multi-ranks mode, please install oneCCL firstly.
- [oneCCL Installation](https://github.com/oneapi-src/oneCCL)
  - If you have built xfastertransformer from source, oneCCL is installed in 3rdparty when compilation.
    ```
    source ./3rdparty/oneccl/build/_install/env/setvars.sh
    ```
  - ***[Recommended]*** Use provided scripts to build it from source code. 
    ```bash
    cd 3rdparty
    sh prepare_oneccl.sh
    source ./oneccl/build/_install/env/setvars.sh
    ```
  - Install oneCCL through installing [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).***(Notice:It is recommended to use versions 2023.x and below.)*** And source the enviroment by:
    ```
    source /opt/intel/oneapi/setvars.sh
    ```

- Here is a example on local. 
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
model = xfastertransformer.AutoModel.from_pretrained("/data/chatglm-6b-xft", dtype="bf16")

# Slave
while True:
    model.generate()
```
##### C++
`model.getRank()` can get the process's rank, `model.getRank() == 0` is the Master.  
For Slaves, any value can be input to `model.config()` and `model.input` since Master's value will be synced.
```C++
xft::AutoModel model("/data/chatglm-6b-xft", xft::DataType::bf16);

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
  ***PS: Due to the potential compatibility issues between the model file and the `transformers` version, please select the appropriate `transformers` version.***
- Run the script corresponding to the model. After the web server started, open the output URL in the browser to use the demo. Please specify the paths of model and tokenizer directory, and data type. `transformer`'s tokenizer is used to encode and decode text so `${TOKEN_PATH}` means the huggingface model directory. This demo also support multi-rank.
```bash
# Recommend preloading `libiomp5.so` to get a better performance.
# `libiomp5.so` file will be in `3rdparty/mklml/lib` directory after build xFasterTransformer.
LD_PRELOAD=libiomp5.so python examples/web_demo/ChatGLM.py \
                                    --dtype=bf16 \
                                    --token_path=${TOKEN_PATH} \
                                    --model_path=${MODEL_PATH}
```

## Serving
[A example serving of MLServer](serving/mlserver/README.md) is provided which supports REST and gRPC interface and adaptive batching feature to group inference requests together on the fly.

## [Benchmark](benchmark/README.md)

Benchmark scripts are provided to get the model inference performance quickly.
- [Prepare the model](#prepare-model).
- Install the dependencies, including oneCCL and python dependencies.
- Enter the `benchmark` folder and run `run_benchmark.sh`. Please refer to [Benchmark README](benchmark/README.md) for more information.

**Notes!!!**: The system and CPU configuration may be different. For the best performance, please try to modify OMP_NUM_THREADS, datatype and the memory nodes number (check the memory nodes using `numactl -H`) according to your test environment.

## Support

- xFasterTransformer email: xft.maintainer@intel.com
- xFasterTransformer [wechat](https://github.com/intel/xFasterTransformer/wiki)

## Q&A

- ***Q***: Can xFasterTransformer run on a Intel® Core™ CPU?  
***A***: No. xFasterTransformer requires support for the AMX and AVX512 instruction sets, which are not available on Intel® Core™ CPUs.

- ***Q***: Can xFasterTransformer run on the Windows system?  
***A***: There is no native support for Windows, and all compatibility tests are only conducted on Linux, so Linux is recommended.

- ***Q***: Why does the program freeze or exit with errors when running in multi-rank mode after installing the latest version of oneCCL through oneAPI?  
***A***: Please try downgrading oneAPI to version 2023.x or below, or use the provided script to install oneCCL from source code.

- ***Q***: Why does running the program using two CPU sockets result in much lower performance compared to running on a single CPU socket?  
***A***: Running in this way causes the program to engage in many unnecessary cross-socket communications, significantly impacting performance. If there is a need for cross-socket deployment, consider running in a multi-rank mode with one rank on each socket.

- ***Q***:The performance is normal when running in a single rank, but why is the performance very slow and the CPU utilization very low when using MPI to run multiple ranks?   
***A***:This is because the program launched through MPI reads `OMP_NUM_THREADS=1`, which cannot correctly retrieve the appropriate value from the environment. It is necessary to manually set the value of `OMP_NUM_THREADS` based on the actual situation.

- ***Q***: Why do I still encounter errors when converting already supported models?  
***A***: Try downgrading `transformer` to an appropriate version, such as the version specified in the `requirements.txt`. This is because different versions of Transformer may change the names of certain variables.
