# xFasterTransformer

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

xFasterTransformer为大语言模型（LLM）在CPU X86平台上的部署提供了一种深度优化的解决方案，支持多CPU节点之间的分布式部署方案，使得超大模型在CPU上的部署成为可能。此外，xFasterTransformer提供了C++和Python两种API接口，涵盖了从上层到底层的接口调用，易于用户使用并将xFasterTransformer集成到自有业务框架中。

## 目录
- [xFasterTransformer](#xfastertransformer)
  - [目录](#目录)
  - [模型概览](#模型概览)
    - [支持的模型](#支持的模型)
    - [支持的数据类型](#支持的数据类型)
  - [目录](#目录-1)
  - [安装](#安装)
    - [使用 PyPI](#使用-pypi)
    - [使用 Docker](#使用-docker)
    - [从源码构建](#从源码构建)
      - [准备环境](#准备环境)
        - [手动操作](#手动操作)
        - [如何编译](#如何编译)
  - [模型准备](#模型准备)
  - [API 用法](#api-用法)
    - [Python API(PyTorch)](#python-apipytorch)
    - [C++ API](#c-api)
  - [如何运行](#如何运行)
    - [单进程](#单进程)
    - [多进程](#多进程)
      - [命令行调用](#命令行调用)
      - [代码实现](#代码实现)
        - [Python](#python)
        - [C++](#c)
  - [网页示例](#网页示例)
  - [服务](#服务)
  - [性能测试](#性能测试)
  - [技术支持](#技术支持)
  - [问题与回答](#问题与回答)

## 模型概览
大型语言模型（LLM）的发展速度非常快，在许多人工智能场景中得到了广泛的应用。xFasterTransformer 充分利用了至强平台的硬件能力，在单颗CPU和多颗CPU/多节点上实现了 LLM 推理的高性能和高可扩展性。

xFasterTransformer 提供了一系列 C++ 和 Python 应用程序接口，终端用户可将 xFasterTransformer 直接集成到自己的解决方案或服务中。此外，xFT还提供了多种示例代码来演示使用方法。包括供用户进行性能测试的测试代码和脚本，以及通过网页模式搭建常用 LLM 模型服务的示例。


### 支持的模型

|        模型        |   框架   |          | 分布式支持 |
| :----------------: | :------: | :------: | :--------: |
|                    | PyTorch  |   C++    |            |
|      ChatGLM       | &#10004; | &#10004; |  &#10004;  |
|      ChatGLM2      | &#10004; | &#10004; |  &#10004;  |
|      ChatGLM3      | &#10004; | &#10004; |  &#10004;  |
|       Llama        | &#10004; | &#10004; |  &#10004;  |
|       Llama2       | &#10004; | &#10004; |  &#10004;  |
|      Baichuan      | &#10004; | &#10004; |  &#10004;  |
|        QWen        | &#10004; | &#10004; |  &#10004;  |
| SecLLM(YaRN-Llama) | &#10004; | &#10004; |  &#10004;  |
|        Opt         | &#10004; | &#10004; |  &#10004;  |
|   Deepseek-coder   | &#10004; | &#10004; |  &#10004;  |
|       gemma        | &#10004; | &#10004; |  &#10004;  |
|     gemma-1.1      | &#10004; | &#10004; |  &#10004;  |
|     codegemma      | &#10004; | &#10004; |  &#10004;  |

### 支持的数据类型

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

## 目录
xFasterTransformer 文档和[Wiki](https://github.com/intel/xFasterTransformer/wiki)提供了以下资源：
- xFasterTransformer 简介。
- C++ 和 PyTorch 上层和底层接口的全面 API 参考资料。
- 在 C++ 和 PyTorch 中使用 xFasterTransformer 的实用 API 示例。

## 安装
### 使用 PyPI
```bash
pip install xfastertransformer
```

### 使用 Docker
```bash
docker pull intel/xfastertransformer:latest
```
使用命令运行 docker (假设模型文件位于 `/data/` 目录):  
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
**注意!!!**: 如果在多进程模式下运行时发生**bus error**，请增大"--shm-size"。docker默认限制共享内存大小为64MB，而我们的实现使用大量的共享内存来获得更好的性能。

### 从源码构建
#### 准备环境
##### 手动操作
- [PyTorch](https://pytorch.org/get-started/locally/) v2.0 (使用 PyTorch API 时需要，但使用 C++ API 时不需要。)
  ```bash 
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

##### 安装依赖的库

请安装所依赖的libnuma库:
- CentOS: yum install libnuma-devel
- Ubuntu: apt-get install libnuma-dev


##### 如何编译
- 使用 'CMake'
  ```bash
  # 构建 xFasterTransformer
  git clone https://github.com/intel/xFasterTransformer.git xFasterTransformer
  cd xFasterTransformer
  git checkout <latest-tag>
  # 如果使用python示例，请确保已经安装torch。
  mkdir build && cd build
  cmake ..
  make -j
  ```
- 使用 `python setup.py`
  ```bash
  # 构建Build xFasterTransformer库和C++示例。
  python setup.py build

  # 安装xFastertransformer到pip环境中。
  # 注意：在安装之前请运行 `python setup.py build`！
  python setup.py install
  ```

## [模型准备](tools/README.md)
xFasterTransformer 支持的模型格式与 Huggingface 有所不同，但与 FasterTransformer 的格式兼容。
1. 首先下载 huggingface 格式的模型。
2. 然后，使用 xfastertransformer 中的模型转换模块将模型转换为 xFasterTransformer 格式。如果没有提供输出目录，转换后的模型将被默认放置到 `${HF_DATASET_DIR}-xft`.
    ```
    python -c 'import xfastertransformer as xft; xft.LlamaConvert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
    ```
    ***PS: 由于模型文件和 `transformers` 版本之间可能存在兼容性问题，请选择相应的 `transformers` 版本。***
    
    支持的模型转换列表：
    - LlamaConvert
    - YiConvert
    - GemmaConvert
    - ChatGLMConvert
    - ChatGLM2Convert
    - ChatGLM3Convert
    - OPTConvert
    - BaichuanConvert
    - QwenConvert
    - DeepseekConvert

## API 用法
更多详情，请参阅 API 文档和 [示例](examples/README.md).
### Python API(PyTorch)
首先，请安装依赖项。
- Python 依赖项
  ```bash
  pip install -r requirements.txt
  ```
  ***PS: 由于模型文件和 `transformers`版本之间可能存在兼容性问题，请选择适当的 `transformers`版本。***
- oneCCL (用于多进程)  
  安装 oneCCL 并设置环境。请参阅[准备环境](#prepare-environment).


xFasterTransformer 的 Python API 与transformers类似，也支持transformers的streamer来实现流式输出。在示例中，我们使用transformers将输入文字进行编码，生成token id。
```Python
import xfastertransformer
from transformers import AutoTokenizer, TextStreamer
# 假设huggingface格式的模型目录为`/data/chatglm-6b-hf`，转换后模型的目录为`/data/chatglm-6b-xft`.
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
[SentencePiece](https://github.com/google/sentencepiece) 可用于文本编码和解码。
```C++
#include <vector>
#include <iostream>
#include "xfastertransformer.h"
// ChatGLM token ids for prompt "Once upon a time, there existed a little girl who liked to have adventures."
std::vector<int> input(
        {3393, 955, 104, 163, 6, 173, 9166, 104, 486, 2511, 172, 7599, 103, 127, 17163, 7, 130001, 130004});

// 假设转换后的模型目录为`/data/chatglm-6b-xft`.
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

## 如何运行
建议预加载 `libiomp5.so` 以获得更好的性能。成功编译 xFasterTransformer 后，`libiomp5.so` 文件将位于 `3rdparty/mklml/lib` 目录中。
### 单进程
xFasterTransformer 会自动检查 MPI 环境，或者使用 `SINGLE_INSTANCE=1` 环境变量强制停用 MPI。 

### 多进程
#### 命令行调用
使用 MPI 在多进程模式下运行，请先安装 oneCCL。
- [oneCCL 安装](https://github.com/oneapi-src/oneCCL)
  - 如果您从源代码编译了 xfastertransformer，则在编译时会在3rdparty目录安装 oneCCL。
    ```
    source ./3rdparty/oneccl/build/_install/env/setvars.sh
    ```
  - ***[推荐]*** 使用提供的脚本从源代码中构建。
    ```bash
    cd 3rdparty
    sh prepare_oneccl.sh
    source ./oneccl/build/_install/env/setvars.sh
    ```
  - 通过 [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)安装 oneCCL。***(注意：建议使用 2023.x 及以下版本。)*** 并通过以下方式提供环境:
    ```
    source /opt/intel/oneapi/setvars.sh
    ```

- 下面是一个本地环境的运行方式示例。 
  ```bash
  OMP_NUM_THREADS=48 LD_PRELOAD=libiomp5.so mpirun \
    -n 1 numactl -N 0  -m 0 ${RUN_WORKLOAD} : \
    -n 1 numactl -N 1  -m 1 ${RUN_WORKLOAD} 
  ```

#### 代码实现
更多详情，请参阅示例。
##### Python
`model.rank` 可以获得进程的编号，`model.rank == 0` 是主进程。 
对于从属进程，加载模型后只需要做 `model.generate()`。输入和生成配置将自动同步。
```Python
model = xfastertransformer.AutoModel.from_pretrained("/data/chatglm-6b-xft", dtype="bf16")

# Slave
while True:
    model.generate()
```
##### C++
`model.getRank()`可以获取进程的编号，`model.getRank() == 0` 是主进程。 
对于从属进程，可以向 `model.config()` 和 `model.input` 输入任何值，因为主进程的值将被同步。
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

## [网页示例](examples/web_demo/README.md)
本仓库中提供了基于 [Gradio](https://www.gradio.app/)的网页demo。现在支持 ChatGLM、ChatGLM2 和 Llama2 模型。
- [准备模型](#prepare-model).
- 安装依赖项
  ```bash
  pip install -r examples/web_demo/requirements.txt
  ```
  ***PS: 由于模型文件和 `transformers`版本之间可能存在兼容性问题，请选择适当的 `transformers`版本。***
- 运行与模型相对应的脚本。网络服务器启动后，在浏览器中打开输出 URL 以使用演示程序。请指定模型和tokenizer目录的路径以及数据类型。`transformer`的tokenizer用于对文本进行编码和解码，因此`${TOKEN_PATH}`指的是 huggingface 模型目录。此演示还支持多进程。
```bash
# 推荐预加载`libiomp5.so`来获得更好的性能。
# `libiomp5.so`文件会位于编译后`3rdparty/mklml/lib`文件夹中。
LD_PRELOAD=libiomp5.so python examples/web_demo/ChatGLM.py \
                                    --dtype=bf16 \
                                    --token_path=${TOKEN_PATH} \
                                    --model_path=${MODEL_PATH}
```

## 服务
[MLServer 服务示例](serving/mlserver/README.md) 支持 REST 和 gRPC 接口，并具有自适应批处理功能，可即时将推理请求分组。

## [性能测试](benchmark/README.md)

提供的Benchmark脚本可快速获得模型推理性能。
- [准备模型](#prepare-model).
- 安装依赖项，包括 oneCCL 和 python 依赖项。
- 进入 `benchmark` 文件夹并运行 `run_benchmark.sh`。更多信息请参阅 [Benchmark README](benchmark/README.md)。

**备注!!!**: 系统和 CPU 配置可能不同。为获得最佳性能，请尝试根据测试环境修改 OMP_NUM_THREADS、数据类型和内存节点数（使用 `numactl -H` 检查内存节点）。

## 技术支持

- xFasterTransformer 邮件: xft.maintainer@intel.com
- xFasterTransformer [微信](https://github.com/intel/xFasterTransformer/wiki)

## 问题与回答

- ***问***: xFasterTransformer 可以在 Intel® Core™ CPU 上运行吗？   
***答***: 不可以。xFasterTransformer 需要 AMX 和 AVX512 指令集的支持，而Intel® Core™ CPU不支持这些指令集。

- ***问***: xFasterTransformer 可以在 Windows 系统上运行吗？  
***答***: 不支持 Windows，所有兼容性测试都只在 Linux 上进行，因此建议使用 Linux。

- ***问***: 通过 oneAPI 安装了最新版本的 oneCCL 后，在多进程模式下运行时，为什么程序会卡死或出错？  
***答***: 请尝试将 oneAPI 降级到 2023.x 或更低版本，或使用提供的脚本从源代码安装 oneCCL。

- ***问***: 为什么使用两个 CPU 运行程序的性能比使用单个 CPU 运行程序的性能要低得多？  
***答***: 以这种方式运行会导致程序进行许多不必要的跨CPU通信，严重影响性能。如果需要跨CPU部署，可考虑在多进程模式下运行，在每个CPU上部署一个进程。

- ***问***:以单进程运行时性能正常，但为什么使用 MPI 运行多进程性能很慢，CPU 利用率很低？   
***答***:这是因为通过 MPI 启动的程序读取的是 `OMP_NUM_THREADS=1`，无法从环境中正确获取相应的值。有必要根据实际情况手动设置 `OMP_NUM_THREADS` 的值。

- ***问***: 为什么在转换已支持的模型时仍会遇到错误？  
***答***: 尝试将 `transformer` 降级到合适的版本，如 `requirements.txt` 中指定的版本。这是因为不同版本的 Transformer 可能会更改某些变量的名称。
