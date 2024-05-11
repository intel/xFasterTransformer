
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

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
CORES=`lscpu | grep Core | awk '{print $4}'`
export OMP_NUM_THREADS=$CORES

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

numactl -N 0 python training/finetune.py $ARGS \
    --base_model ${MODEL_PATH} \
    --data_path 'alpaca_data.json' \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
    #--max_steps ${MAXSTEP}

train_samples_per_second=($(grep -i 'train_samples_per_second'  ${OUTPUT_DIR}/training_log_${precision}_${mode}* |sed -e 's/.*train_samples_per_second*//;s/[^0-9.,]//g;' | awk -F, '{print $1}' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
train_loss=($(grep -i 'train_loss' ${OUTPUT_DIR}/training_log_${precision}_${mode}* |sed -e 's/.*train_loss*//;s/[^0-9.,]//g;' | awk -F, '{print $1}' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
echo ""LLaMa";training throughput;"train_samples_per_second";${precision};${BATCH_SIZE}; ${train_samples_per_second} " |tee -a ${OUTPUT_DIR}/summary.log
echo ""LLaMa";training throughput;"train_loss";${precision};${BATCH_SIZE}; ${train_loss} " |tee -a ${OUTPUT_DIR}/summary.log

