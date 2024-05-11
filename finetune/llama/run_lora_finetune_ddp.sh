#!/bin/bash

#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ARGS=""

MAXSTEP=${MAXSTEP:-50}

export LD_PRELOAD=../../3rdparty/mklml/lib/libiomp5.so
export LD_PRELOAD=../../3rdparty/jemalloc/install_dir/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
if [ -z "${MODEL_PATH}" ]; then
  echo "The required environment variable MODEL_PATH has not been set, please set Llama2-7b model path to MODEL_PATH"
  exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16 "
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16 "
    echo "### running fp16 mode"
elif [[ "$1" == "bf32" ]]
then
    precision=bf32
    ARGS="$ARGS --bf32 "
    echo "### running bf32 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
NNODES=${NNODES:-1}
HOSTFILE=${HOSTFILE:-./hostfile}
NUM_RANKS=$(( NNODES * SOCKETS ))

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
#oneCCL settings
export I_MPI_PIN_DOMAIN=[0xffffffffffff,0xffffffffffff000000000000]
export CCL_WORKER_COUNT=8
export CCL_LOG_LEVEL=info
export CCL_BF16=avx512bf
export CCL_ATL_TRANSPORT=mpi
#export CCL_ATL_TRANSPORT=ofi
export CCL_MNIC_COUNT=2
export CCL_MNIC=local
export CCL_MNIC_NAME=irdma1,irdma5
export CCL_ALLREDUCE=ring

for (( i = $SOCKETS; i < 2*$SOCKETS; i++ )); do  # pin CCL workers to HT
  START_CORE=$(( i * CORES ))
  for (( j = 0; j < $CCL_WORKER_COUNT; j++)); do
   CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((START_CORE + j))"
  done
done

export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`


#DDP settings
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_ADDR=`head -1 hostfile`
export MASTER_PORT=29500

# Fabric settings
#export FI_PROVIDER=psm3
export FI_PROVIDER=tcp
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_PRINT_STATS=0
export PSM3_RV_MR_CACHE_SIZE=8192
export PSM3_KASSIST_MODE=none
#export PSM3_NIC='irdma*
export FI_PSM3_CONN_TIMEOUT=100

oneccl_bindings_for_pytorch_path=../../3rdparty/oneCCL/build/_install
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

#export FI_PROVIDER_PATH=$oneccl_bindings_for_pytorch_path/lib/prov
mpiexec.hydra -l -np ${NUM_RANKS} -ppn ${SOCKETS} \
    -genv KMP_AFFINITY=${KMP_AFFINITY} \
    -genv KMP_BLOCKTIME=${KMP_BLOCKTIME} \
    -genv OMP_NUM_THREADS=${CORES_PER_INSTANCE} \
    -genv MASTER_ADDR=${MASTER_ADDR} \
    -genv MASTER_PORT=${MASTER_PORT} \
    -genv I_MPI_PIN_DOMAIN=${I_MPI_PIN_DOMAIN} \
    -genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT} \
    -genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY} \
    python -u training/finetune.py $ARGS \
    --base_model ${MODEL_PATH} \
    --data_path 'alpaca_data.json' \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --micro_batch_size 8 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --ddp_backend ccl