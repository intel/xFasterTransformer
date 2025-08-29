# xFT Accuracy Evalution with opencompass
OpenCompass is an LLM evaluation platform, supporting a wide range of models (InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets. more details information can refer to [https://opencompass.org.cn/](https://opencompass.org.cn/)

## Installation
Below are the steps for quick installation and datasets preparation.

### Environment Setup
``` bash
# setup steps is refer to https://opencompass.org.cn/doc
$ conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
$ conda activate opencompass

$ git clone -b intel/xft https://github.com/marvin-Yu/opencompass.git && cd opencompass
$ pip install -e .
```

### Data Preparation
``` bash
# download core dataset
$ wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip

$ unzip OpenCompassData-core-20240207.zip

# # download full dataset
# $ wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# $ unzip OpenCompassData-complete-20240207.zip
# $ cd ./data
# $ find . -name "*.zip" -exec unzip "{}" \;
```

### Model Preparation
Download model weights to the `/data/models` directory (path configuration is not supported temporarily; you can modify the configuration file or create symbolic links). For example, to test the model `chatglm2_6b`:
```bash
/data/models/
├── chatglm2-6b-hf
├── chatglm2-6b-hf-xft
├── ...
```
For exporting xFT models, please refer to [xFT Models Preparation](https://github.com/intel/xFasterTransformer?tab=readme-ov-file#models-preparation).

### xFT Evaluation
``` bash
# list all xFT support models
$ python tools/list_configs.py xft
# +------------------------+----------------------------------------------+
# | Model                  | Config Path                                  |
# |------------------------+----------------------------------------------|
# | xft_llama2_13b_chat    | configs/models/xft/xft_llama2_13b_chat.py    |
# | xft_llama2_70b_chat    | configs/models/xft/xft_llama2_70b_chat.py    |
# | xft_llama2_7b_chat     | configs/models/xft/xft_llama2_7b_chat.py     |
# | xft_chatglm2_6b        | configs/models/xft/xft_chatglm2_6b.py        |
# | xft_chatglm3_6b        | configs/models/xft/xft_chatglm3_6b.py        |
# | xft_chatglm_6b         | configs/models/xft/xft_chatglm_6b.py         |
# | xft_gemma_2b_it        | configs/models/xft/xft_gemma_2b_it.py        |
# | xft_gemma_7b_it        | configs/models/xft/xft_gemma_7b_it.py        |
# | ...............        | .....................................        |
# +------------------------+----------------------------------------------+

# list dataset than you want.
$ python tools/list_configs.py ceval
# +--------------------------------+------------------------------------------------------------------+
# | Dataset                        | Config Path                                                      |
# |--------------------------------+------------------------------------------------------------------|
# | ceval_gen                      | configs/datasets/ceval/ceval_gen.py                              |
# | ceval_gen_5f30c7               | configs/datasets/ceval/ceval_gen_5f30c7.py                       |
# | ceval_ppl                      | configs/datasets/ceval/ceval_ppl.py                              |
# | ceval_ppl_93e5ce               | configs/datasets/ceval/ceval_ppl_93e5ce.py                       |
# | ...............                | .............................................                    |
# +--------------------------------+------------------------------------------------------------------+

# run eval test
$ python run.py --max-num-workers 1 --models xft_chatglm2_6b --datasets ceval_gen
# 20240416_100621
# tabulate format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset                                         version    metric         mode     xft_chatglm2_6b-bf16   xft_chatglm2_6b-xft-bf16
# ----------------------------------------------  ---------  -------------  ------  ---------------------  -------------------------
# ceval-computer_network                          db9ce2     accuracy       gen                     47.37                      47.37
# ceval-operating_system                          1c2571     accuracy       gen                     xx.xx                      xx.xx
# ceval-computer_architecture                     a74dad     accuracy       gen                     xx.xx                      xx.xx
# ceval-college_programming                       4ca32a     accuracy       gen                     xx.xx                      xx.xx
# ceval-college_physics                           963fa8     accuracy       gen                     xx.xx                      xx.xx
# ceval-college_chemistry                         e78857     accuracy       gen                     xx.xx                      xx.xx
# ceval-advanced_mathematics                      ce03e2     accuracy       gen                     xx.xx                      xx.xx
# ...
```

# FAQ

### AttributeError: 'ChatGLMTokenizer' object has no attribute 'tokenizer'?
`pip install --force-reinstall transformers==4.33.0`

### for `TruthfulQA` dataset, pls install deps with `bleurt`
`pip install git+https://github.com/google-research/bleurt.git`

### add env macro to control the test case, (XFT_ONLY_XFT & XFT_DTYPE_LIST & XFT_KVCACHE_DTYPE_LIST)
```
# XFT_ONLY_XFT
# This environment variable is used exclusively for testing with XFT, without testing with HF models.

# XFT_DTYPE_LIST contains a list like:
# [
#   "fp16",
#   "bf16",
#   "int8",
#   "w8a8",
#   "int4",
#   "nf4",
#   "bf16_fp16",
#   "bf16_int8",
#   "bf16_w8a8",
#   "bf16_int4",
#   "bf16_nf4",
#   "w8a8_int8",
#   "w8a8_int4",
#   "w8a8_nf4",
# ]
# For example, it can include a single parameter like XFT_DTYPE_LIST=bf16
# or multiple parameters separated by commas like XFT_DTYPE_LIST=bf16,fp16,int8.

# XFT_KVCACHE_DTYPE_LIST environment variable contains a list of data types used for XFT KV cache.
# [
#   "fp32",
#   "fp16",
#   "int8",
# ]
# For example, it can include a single parameter like XFT_KVCACHE_DTYPE_LIST=fp16
# or multiple parameters separated by commas like XFT_KVCACHE_DTYPE_LIST=fp32,fp16,int8.
```