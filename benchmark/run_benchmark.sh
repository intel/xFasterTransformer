#!/bin/bash
# Copyright (c) 2023 Intel Corporation
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
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In this benchmark case, precision can be: bf16, bf16_fp16, fp16, bf16_int8, int8 
precision="fp16"
# The model name can be choosen are: llama-2(-7b,-13b,-70b), llama(-7b,-13b,-30b,-65b), chatglm2-6b, chatglm-6b, baichuan2(-7b,13b)
# which you can refer to ../examples/model_config as well
model_name="chatglm2-6b"

# Example here is using fake model, you can use real model as well
export XFT_FAKE_MODEL=1
model_path="${SCRIPT_DIR}"/../examples/model_config/${model_name}/

benchmark_cmd="python "${SCRIPT_DIR}"/benchmark.py \
    --token_path "${model_path}" \
    --model_path "${model_path}" \
    --prompt_path "${SCRIPT_DIR}"/prompt.json \
    --model_name "${model_name}" \
    --dtype "${precision}" \
    --token_in 32 	\
    --token_out 32 --beam_width 1 --iteration 10"

if [[ ${model_name} == *"llama"* ]] || [[ ${model_name} == *"baichuan-"* ]]; then
    benchmark_cmd+=" --padding=False"
fi

sockets=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
cores_per_socket=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
numa_nodes=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}'`
cores_per_numa=$(( $sockets * $cores_per_socket / $numa_nodes ))

if [ "${numa_nodes}" -eq 16 ]; then
    #HBM flat SNC-4 mode, Confirm that there are 8 HBM memory nodes and 8 DRAM memory nodes through "numactl -H"
    #0-7 is DRAM memory node, 8-15 is HBM node
    export OMP_NUM_THREADS=$((${cores_per_numa} * 2))
    echo "HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 numactl -p 8  -N 0 ${benchmark_cmd}   : \
    -n 1 numactl -p 9  -N 1 ${benchmark_cmd}  : \
    -n 1 numactl -p 10 -N 2 ${benchmark_cmd} : \
    -n 1 numactl -p 11 -N 3 ${benchmark_cmd} "
elif [ "${numa_nodes}" -eq 8 ]; then
    #HBM SNC-4 for cache or hbm only mode
    export OMP_NUM_THREADS=${cores_per_numa}
    echo "HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 numactl -m 0  -N 0 ${benchmark_cmd}   : \
    -n 1 numactl -m 1  -N 1 ${benchmark_cmd}  : \
    -n 1 numactl -m 2  -N 2 ${benchmark_cmd} : \
    -n 1 numactl -m 3  -N 3 ${benchmark_cmd} "
elif [ "${numa_nodes}" -eq 4 ]; then
    #HBM flat Quad-mode, Confirm that there are 2 HBM memory nodes and 2 DRAM memory nodes through "nuamctl -H"
    echo "HBM Quad mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} * 2))
    run_cmd="mpirun \
    -n 1 numactl -N 0 -p 2 ${benchmark_cmd} : \
    -n 1 numactl -N 1 -p 3 ${benchmark_cmd} "
elif [ "${numa_nodes}" -eq 2 ]; then
    #SPR or hbm only or hbm cache Quad-mode, Confirm that there are 2 DRAM memory nodes through "nuamctl -H"
    echo "SPR Quad mode"
    export OMP_NUM_THREADS=${cores_per_numa}
    run_cmd="mpirun \
    -n 1 numactl -N 0  -m 0 ${benchmark_cmd}"
else
    echo "Please double check the memory nodes"
fi

echo "Run command line: ${run_cmd}"
eval ${run_cmd}

# In this benchmark case, token_in only can be "demo","32","64","128","256","512","1024","2016"
# "32" means the token length is 32, if needs more test, add it into input_token.py