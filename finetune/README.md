# PyTorch LLM finetuning guide

## Description
This document is a guide for running LLM LoRA finetuning using PyTorch on CPU.
Both single socket and multi-ranks distributed are supported.

## Step-by-step run guide
### Prepare dependency
Recommended to use Python 3.9 or higher version.
```
pip install -r requirements.txt
# To use oneccl as the distributed backend in distributed training on CPU.
python -m pip install oneccl_bind_pt==2.0.0 -f https://developer.intel.com/ipex-whl-stable-cpu
```
### Install jemalloc (Optional)
Install jemalloc either using conda or from source.

Using conda:
```
conda install jemalloc
```
From source:
```
cd ../3rdparty
git clone https://github.com/jemalloc/jemalloc.git 
cd jemalloc
git checkout c8209150f9d219a137412b06431c9d52839c7272
./autogen.sh
./configure --prefix=your_absolute_path(e.g. /home/xxx/xFasterTransformer/3rdparty/jemalloc/install_dir)
make
make install
cd ../finetune/llama
```
## Quick Start Scripts (single socket)
### Env vars
```
export MODEL_PATH=<path to model>
export OUTPUT_DIR=<path to an output directory>
```
### Run script
| DataType    | Run command |
| ----------- | ----------- |
|  BF16       | bash run_lora_finetune.sh bf16 |
|  FP16       | bash run_lora_finetune.sh fp16 |
|  FP32       | bash run_lora_finetune.sh fp32 |
|  BF32       | bash run_lora_finetune.sh bf32 |
## Quick Start Scripts (distributed)
### Env vars
```
# The NNODES is the number of ip in the HOSTFILE, default using 1 node for single-node multi-sockets
export NNODES=#your_node_number
```
create your_ip_list_file, one ip per line
```
scontrol show hostname > ./hostfile
export HOSTFILE=hostfile 
export MODEL_PATH=<path to model>
export OUTPUT_DIR=<path to an output directory>
```
### Run script
| DataType    | Run command |
| ----------- | ----------- |
|  BF16       | bash run_lora_finetune_ddp.sh bf16 |
|  FP16       | bash run_lora_finetune_ddp.sh fp16 |
|  FP32       | bash run_lora_finetune_ddp.sh fp32 |
|  BF32       | bash run_lora_finetune_ddp.sh bf32 |
