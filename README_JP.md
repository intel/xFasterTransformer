# xFasterTransformer

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

xFasterTransformerは、GPUプラットフォーム上のFasterTransformerと同様に、X86プラットフォーム上の大規模言語モデル（LLM）に対して非常に最適化されたソリューションです。xFasterTransformerは、複数のソケットやノードにまたがって分散モードで動作し、より大きなモデルの推論をサポートします。さらに、高レベルから低レベルのインターフェースにわたるC++およびPython APIの両方を提供し、採用と統合が容易です。

## 目次
- [xFasterTransformer](#xfastertransformer)
  - [目次](#目次)
  - [モデルの概要](#モデルの概要)
    - [サポートされているモデル](#サポートされているモデル)
    - [サポートされているデータ型](#サポートされているデータ型)
  - [ドキュメント](#ドキュメント)
  - [インストール](#インストール)
    - [PyPIからインストール](#pypiからインストール)
    - [Dockerを使用](#dockerを使用)
    - [ソースからビルド](#ソースからビルド)
      - [環境の準備](#環境の準備)
        - [手動での準備](#手動での準備)
        - [依存ライブラリのインストール](#依存ライブラリのインストール)
        - [ビルド方法](#ビルド方法)
  - [モデルの準備](#モデルの準備)
  - [APIの使用方法](#apiの使用方法)
    - [Python API(PyTorch)](#python-apipytorch)
    - [C++ API](#c-api)
  - [実行方法](#実行方法)
    - [シングルランク](#シングルランク)
    - [マルチランク](#マルチランク)
      - [コマンドライン](#コマンドライン)
      - [コード](#コード)
        - [Python](#python)
        - [C++](#c)
  - [Webデモ](#webデモ)
  - [サービング](#サービング)
    - [vLLM](#vllm)
      - [インストール](#インストール)
      - [OpenAI互換サーバー](#openai互換サーバー)
    - [FastChat](#fastchat)
    - [MLServer](#mlserver)
  - [ベンチマーク](#ベンチマーク)
  - [サポート](#サポート)
  - [受理された論文](#受理された論文)
  - [Q\&A](#qa)

## モデルの概要
大規模言語モデル（LLM）は非常に速く発展し、多くのAIシナリオで広く使用されています。xFasterTransformerは、主流で人気のあるLLMモデルを使用して、Xeon上でのLLM推論のための最適化されたソリューションです。xFasterTransformerは、Xeonプラットフォームのハードウェア機能を最大限に活用し、シングルソケットおよび複数ソケット/複数ノードでのLLM推論の高性能と高スケーラビリティを実現します。

xFasterTransformerは、エンドユーザーが自分のソリューションやサービスに直接xFasterTransformerを統合できるようにするための一連のAPI（C++およびPythonの両方）を提供します。使用方法を示すためのさまざまな種類のサンプルコードも提供されています。ベンチマークコードとスクリプトは、ユーザーがパフォーマンスを示すために提供されています。人気のあるLLMモデルのWebデモも提供されています。

### サポートされているモデル

|       モデル       | フレームワーク |          | 分散 |
| :----------------: | :-------: | :------: | :----------: |
|                    |  PyTorch  |   C++    |              |
|      ChatGLM       | &#10004;  | &#10004; |   &#10004;   |
|      ChatGLM2      | &#10004;  | &#10004; |   &#10004;   |
|      ChatGLM3      | &#10004;  | &#10004; |   &#10004;   |
|        GLM4        | &#10004;  | &#10004; |   &#10004;   |
|       Llama        | &#10004;  | &#10004; |   &#10004;   |
|       Llama2       | &#10004;  | &#10004; |   &#10004;   |
|       Llama3       | &#10004;  | &#10004; |   &#10004;   |
|     Baichuan       | &#10004;  | &#10004; |   &#10004;   |
|     Baichuan2      | &#10004;  | &#10004; |   &#10004;   |
|        QWen        | &#10004;  | &#10004; |   &#10004;   |
|        QWen2       | &#10004;  | &#10004; |   &#10004;   |
| SecLLM(YaRN-Llama) | &#10004;  | &#10004; |   &#10004;   |
|        Opt         | &#10004;  | &#10004; |   &#10004;   |
|   Deepseek-coder   | &#10004;  | &#10004; |   &#10004;   |
|      gemma         | &#10004;  | &#10004; |   &#10004;   |
|     gemma-1.1      | &#10004;  | &#10004; |   &#10004;   |
|     codegemma      | &#10004;  | &#10004; |   &#10004;   |

### サポートされているデータ型

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

## ドキュメント
xFasterTransformerのドキュメントと[Wiki](https://github.com/intel/xFasterTransformer/wiki)には、以下のリソースが含まれています：
- xFasterTransformerの紹介。
- C++およびPyTorchの高レベルおよび低レベルのインターフェースに関する包括的なAPIリファレンス。
- C++およびPyTorchでのxFasterTransformerの実用的なAPI使用例。

## インストール
### PyPIからインストール
```bash
pip install xfastertransformer
```

### Dockerを使用
```bash
docker pull intel/xfastertransformer:latest
```
Dockerを以下のコマンドで実行します（モデルファイルが`/data/`ディレクトリにあると仮定します）：  
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
**注意!!!**: マルチランクモードで実行中に**バスエラー**が発生した場合は、`--shm-size`を大きくしてください。デフォルトのDockerは共有メモリサイズを64MBに制限しており、私たちの実装ではより多くの共有メモリを使用してより良いパフォーマンスを実現しています。

### ソースからビルド
#### 環境の準備
##### 手動での準備
- [PyTorch](https://pytorch.org/get-started/locally/) v2.3（PyTorch APIを使用する場合に必要ですが、C++ APIを使用する場合は不要です。）
  ```bash 
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

- GPU用のxFTには、DPC++がABI=1を必要とするため、[torch==2.3.0+cpu.cxx11.abi](https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.3.0%2Bcpu.cxx11.abi-cp38-cp38-linux_x86_64.whl#sha256=c34512c3e07efe9b7fb5c3a918fef1a7c6eb8969c6b2eea92ee5c16a0583fe12)の[torch-whl-list](https://download.pytorch.org/whl/torch/)からABI=1をインストールする必要があります。

##### 依存ライブラリのインストール

libnumaパッケージをインストールしてください：
- CentOS: yum install libnuma-devel
- Ubuntu: apt-get install libnuma-dev

##### ビルド方法
- 'CMake'を使用
  ```bash
  # xFasterTransformerをビルド
  git clone https://github.com/intel/xFasterTransformer.git xFasterTransformer
  cd xFasterTransformer
  git checkout <latest-tag>
  # Pythonの例を実行する場合は、torchがインストールされていることを確認してください
  mkdir build && cd build
  cmake ..
  make -j
  ```
- `python setup.py`を使用
  ```bash
  # xFasterTransformerライブラリとC++の例をビルド
  python setup.py build

  # xFasterTransformerをpip環境にインストール
  # 注意：インストール前に`python setup.py build`を実行してください！
  python setup.py install
  ```

## [モデルの準備](tools/README.md)
xFasterTransformerは、Huggingfaceとは異なるモデル形式をサポートしていますが、FasterTransformerの形式と互換性があります。
1. まず、Huggingface形式のモデルをダウンロードします。
2. その後、xfastertransformerのモデル変換モジュールを使用してモデルをxFasterTransformer形式に変換します。出力ディレクトリが提供されていない場合、変換されたモデルは`${HF_DATASET_DIR}-xft`に配置されます。
    ```
    python -c 'import xfastertransformer as xft; xft.LlamaConvert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
    ```
    ***注意: モデルファイルと`transformers`バージョンの間に互換性の問題がある可能性があるため、適切な`transformers`バージョンを選択してください。***
    
    サポートされているモデル変換リスト：
    - LlamaConvert
    - YiConvert
    - GemmaConvert
    - ChatGLMConvert
    - ChatGLM2Convert
    - ChatGLM4Convert
    - OPTConvert
    - BaichuanConvert
    - Baichuan2Convert
    - QwenConvert
    - Qwen2Convert
    - DeepseekConvert

## APIの使用方法
詳細については、APIドキュメントと[例](examples/README.md)を参照してください。
### Python API(PyTorch)
まず、依存関係をインストールしてください。
- Pythonの依存関係
  ```bash
  pip install -r requirements.txt
  ```
  ***注意: モデルファイルと`transformers`バージョンの間に互換性の問題がある可能性があるため、適切な`transformers`バージョンを選択してください。***
- oneCCL（マルチランク用）  
  oneCCLをインストールし、環境を設定します。[環境の準備](#prepare-environment)を参照してください。

xFasterTransformerのPython APIはtransformersに似ており、transformersのストリーマーをサポートしてストリーミング出力を実現します。例では、transformersを使用して入力プロンプトをトークンIDにエンコードします。
```Python
import xfastertransformer
from transformers import AutoTokenizer, TextStreamer
# Huggingfaceモデルディレクトリが`/data/chatglm-6b-hf`、変換後のモデルディレクトリが`/data/chatglm-6b-xft`であると仮定します。
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
[SentencePiece](https://github.com/google/sentencepiece)を使用してテキストをトークン化およびデトークン化できます。
```C++
#include <vector>
#include <iostream>
#include "xfastertransformer.h"
// "Once upon a time, there existed a little girl who liked to have adventures."のプロンプトに対するChatGLMのトークンID
std::vector<int> input(
        {3393, 955, 104, 163, 6, 173, 9166, 104, 486, 2511, 172, 7599, 103, 127, 17163, 7, 130001, 130004});

// 変換後のモデルディレクトリが`/data/chatglm-6b-xft`であると仮定します。
xft::AutoModel model("/data/chatglm-6b-xft", xft::DataType::bf16);

model.config(/*最大長*/ 100, /*ビーム数*/ 1);
model.input(/*入力トークンID*/ input, /*バッチサイズ*/ 1);

while (!model.isDone()) {
    std::vector<int> nextIds = model.generate();
}

std::vector<int> result = model.finalize();
for (auto id : result) {
    std::cout << id << " ";
}
std::cout << std::endl;
```

## 実行方法
より良いパフォーマンスを得るために`libiomp5.so`をプリロードすることをお勧めします。
- ***[推奨]*** xfastertransformerのPythonホイールパッケージがインストールされている場合は、`export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')`を実行します。
- ソースコードからxFasterTransformerをビルドした場合、ビルドが成功すると`libiomp5.so`ファイルは`3rdparty/mkl/lib`ディレクトリにあります。

### シングルランク
FasterTransformerは自動的にMPI環境をチェックしますが、`SINGLE_INSTANCE=1`環境変数を使用してMPIを強制的に無効にすることもできます。

### マルチランク
#### コマンドライン
MPIを使用してマルチランクモードで実行するには、まずoneCCLをインストールしてください。
- [oneCCLのインストール](https://github.com/oneapi-src/oneCCL)
  - ソースからxfastertransformerをビルドした場合、oneCCLはビルド時に3rdpartyにインストールされます。
    ```
    source ./3rdparty/oneccl/build/_install/env/setvars.sh
    ```
  - ***[推奨]*** 提供されたスクリプトを使用してソースコードからビルドします。
    ```bash
    cd 3rdparty
    sh prepare_oneccl.sh
    source ./oneccl/build/_install/env/setvars.sh
    ```
  - [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)をインストールしてoneCCLをインストールします。***(注意：バージョン2023.x以下を使用することをお勧めします。)*** 環境を設定するには以下を実行します：
    ```
    source /opt/intel/oneapi/setvars.sh
    ```

- ローカル環境での実行例を以下に示します。
  ```bash
  # または手動でLD_PRELOAD=libiomp5.soを設定します
  export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')
  OMP_NUM_THREADS=48 mpirun \
    -n 1 numactl -N 0  -m 0 ${RUN_WORKLOAD} : \
    -n 1 numactl -N 1  -m 1 ${RUN_WORKLOAD} 
  ```

#### コード
詳細については、例を参照してください。
##### Python
`model.rank`はプロセスのランクを取得でき、`model.rank == 0`はマスターです。
スレーブの場合、モデルをロードした後に行うべきことは`model.generate()`だけです。入力と生成の設定は自動的に同期されます。
```Python
model = xfastertransformer.AutoModel.from_pretrained("/data/chatglm-6b-xft", dtype="bf16")

# スレーブ
while True:
    model.generate()
```
##### C++
`model.getRank()`はプロセスのランクを取得でき、`model.getRank() == 0`はマスターです。
スレーブの場合、`model.config()`と`model.input`に任意の値を入力できます。マスターの値が同期されるためです。
```C++
xft::AutoModel model("/data/chatglm-6b-xft", xft::DataType::bf16);

// スレーブ
while (1) {
    model.config();
    std::vector<int> input_ids;
    model.input(/*入力トークンID*/ input_ids, /*バッチサイズ*/ 1);

    while (!model.isDone()) {
        model.generate();
    }
}
```

## [Webデモ](examples/web_demo/README.md)
このリポジトリには、[Gradio](https://www.gradio.app/)に基づいたWebデモが提供されています。現在、ChatGLM、ChatGLM2、およびLlama2モデルをサポートしています。
- [モデルの準備](#prepare-model)。
- 依存関係をインストール
  ```bash
  pip install -r examples/web_demo/requirements.txt
  ```
  ***注意: モデルファイルと`transformers`バージョンの間に互換性の問題がある可能性があるため、適切な`transformers`バージョンを選択してください。***
- モデルに対応するスクリプトを実行します。Webサーバーが起動した後、ブラウザで出力URLを開いてデモを使用します。モデルとトークナイザーディレクトリのパス、およびデータ型を指定してください。`transformer`のトークナイザーはテキストのエンコードとデコードに使用されるため、`${TOKEN_PATH}`はHuggingfaceモデルディレクトリを意味します。このデモはマルチランクもサポートしています。
```bash
# より良いパフォーマンスを得るために`libiomp5.so`をプリロードすることをお勧めします。
# または手動でLD_PRELOAD=libiomp5.soを設定します。`libiomp5.so`ファイルはビルド後に`3rdparty/mkl/lib`ディレクトリにあります。
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')
python examples/web_demo/ChatGLM.py \
                      --dtype=bf16 \
                      --token_path=${TOKEN_PATH} \
                      --model_path=${MODEL_PATH}
```

## サービング
### vLLM
vLLMのフォークが作成され、xFasterTransformerバックエンドが統合され、公式のvLLMのほとんどの機能と互換性があります。詳細については[こちらのリンク](serving/vllm-xft.md)を参照してください。

#### インストール
```bash
pip install vllm-xft
```
***注意: 環境に`vllm-xft`と`vllm`の両方を同時にインストールしないでください。パッケージ名は異なりますが、実際には互いに上書きされます。***

#### OpenAI互換サーバー
***注意: `libiomp5.so`のプリロードが必要です！***
```bash
# 以下のコマンドまたは手動でLD_PRELOAD=libiomp5.soを設定してlibiomp5.soをプリロードします
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tokenizer ${TOKEN_PATH} \
        --dtype bf16 \
        --kv-cache-dtype fp16 \
        --served-model-name xft \
        --port 8000 \
        --trust-remote-code
```
マルチランクモードの場合、`python -m vllm.entrypoints.slave`をスレーブとして使用し、スレーブのパラメータをマスターと一致させてください。
```bash
# 以下のコマンドまたは手動でLD_PRELOAD=libiomp5.soを設定してlibiomp5.soをプリロードします
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

OMP_NUM_THREADS=48 mpirun \
        -n 1 numactl --all -C 0-47 -m 0 \
          python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --tokenizer ${TOKEN_PATH} \
            --dtype bf16 \
            --kv-cache-dtype fp16 \
            --served-model-name xft \
            --port 8000 \
            --trust-remote-code \
        : -n 1 numactl --all -C 48-95 -m 1 \
          python -m vllm.entrypoints.slave \
            --dtype bf16 \
            --model ${MODEL_PATH} \
            --kv-cache-dtype fp16
```

### FastChat
xFasterTransformerは[FastChat](https://github.com/lm-sys/FastChat)の公式推論バックエンドです。詳細については、[FastChatのxFasterTransformer](https://github.com/lm-sys/FastChat/blob/main/docs/xFasterTransformer.md)および[FastChatのサービング](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)を参照してください。

### MLServer
[MLServerのサービング例](serving/mlserver/README.md)が提供されており、RESTおよびgRPCインターフェースをサポートし、推論リクエストを即座にグループ化するための適応バッチ機能を備えています。

## [ベンチマーク](benchmark/README.md)

ベンチマークスクリプトが提供されており、モデル推論のパフォーマンスを迅速に取得できます。
- [モデルの準備](#prepare-model)。
- 依存関係をインストールし、oneCCLおよびPythonの依存関係を含めます。
- `benchmark`フォルダに入り、`run_benchmark.sh`を実行します。詳細については[ベンチマークREADME](benchmark/README.md)を参照してください。

**注意!!!**: システムおよびCPU構成は異なる場合があります。最良のパフォーマンスを得るために、テスト環境に応じてOMP_NUM_THREADS、データ型、およびメモリノード数（`numactl -H`を使用してメモリノードを確認）を変更してください。

## サポート

- xFasterTransformerのメール: xft.maintainer@intel.com
- xFasterTransformerの[WeChat](https://github.com/intel/xFasterTransformer/wiki)

## 受理された論文
- ICLR'2024 on practical ML for limited/low resource settings: [Distributed Inference Performance Optimization for LLMs on CPUs](https://arxiv.org/abs/2407.00029)
- ICML'2024 on Foundation Models in the Wild: [Inference Performance Optimization for Large Language Models on CPUs](https://arxiv.org/abs/2407.07304)
- IEEE ICSESS 2024: All-in-one Approach for Large Language Models Inference

xFTがあなたの研究に役立つ場合は、以下を引用してください：
```latex
@article{he2024distributed,
  title={Distributed Inference Performance Optimization for LLMs on CPUs},
  author={He, Pujiang and Zhou, Shan and Li, Changqing and Huang, Wenhuan and Yu, Weifei and Wang, Duyi and Meng, Chen and Gui, Sheng},
  journal={arXiv preprint arXiv:2407.00029},
  year={2024}
}
```
および
```latex
@inproceedings{he2024inference,
  title={Inference Performance Optimization for Large Language Models on CPUs},
  author={He, Pujiang and Zhou, Shan and Huang, Wenhuan and Li, Changqing and Wang, Duyi and Guo, Bin and Meng, Chen and Gui, Sheng and Yu, Weifei and Xie, Yi},
  booktitle={ICML 2024 Workshop on Foundation Models in the Wild}
}
```

## Q&A

- ***Q***: xFasterTransformerはIntel® Core™ CPUで動作しますか？  
***A***: いいえ。xFasterTransformerはAMXおよびAVX512命令セットのサポートを必要とし、これらの命令セットはIntel® Core™ CPUでは利用できません。

- ***Q***: xFasterTransformerはWindowsシステムで動作しますか？  
***A***: Windowsにはネイティブサポートがなく、すべての互換性テストはLinuxでのみ実施されているため、Linuxをお勧めします。

- ***Q***: 最新バージョンのoneCCLをoneAPI経由でインストールした後、マルチランクモードで実行するとプログラムがフリーズしたりエラーが発生したりするのはなぜですか？  
***A***: oneAPIをバージョン2023.x以下にダウングレードするか、提供されたスクリプトを使用してソースコードからoneCCLをインストールしてみてください。

- ***Q***: 2つのCPUソケットを使用してプログラムを実行すると、1つのCPUソケットで実行する場合に比べてパフォーマンスが大幅に低下するのはなぜですか？  
***A***: この方法で実行すると、プログラムが多くの不要なクロスソケット通信を行い、パフォーマンスに大きな影響を与えます。クロスソケット展開が必要な場合は、各ソケットに1つのランクを配置してマルチランクモードで実行することを検討してください。

- ***Q***: シングルランクで実行するとパフォーマンスは正常ですが、MPIを使用して複数のランクで実行するとパフォーマンスが非常に遅く、CPUの利用率が非常に低いのはなぜですか？   
***A***: これは、MPIを介して起動されたプログラムが`OMP_NUM_THREADS=1`を読み取り、環境から適切な値を正しく取得できないためです。実際の状況に応じて`OMP_NUM_THREADS`の値を手動で設定する必要があります。

- ***Q***: すでにサポートされているモデルを変換する際にエラーが発生するのはなぜですか？  
***A***: `transformer`を適切なバージョンにダウングレードしてみてください。これは、異なるバージョンのTransformerが特定の変数の名前を変更する可能性があるためです。
