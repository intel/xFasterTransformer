#!/bin/bash

# conda create --name xft python=3.8
conda activate xft
source /opt/intel/oneapi/setvars.sh # 2024.0
source ../3rdparty/oneccl/build/_install/env/setvars.sh

export CC=icx
export CXX=icpx
cmake -DBUILD_WITH_SHARED_LIBS=ON ..
make -j

LD_LIBRARY_PATH=/home/xfast/xFasterTransformer/build/ SINGLE_INSTANCE=1 OMP_NUM_THREADS=20 mpirun -n 1 numactl -N 1 -m 1 ./example --model /home/xfast/models/llama-2-7b-chat-xft/ --token /home/xfast/models/llama-2-7b-chat-hf/tokenizer.model --dtype fp16 --loop 3 --no_stream --input_len 16 --output_len 16
