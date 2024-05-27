#!/bin/bash
# Copyright (c) 2023-2024 Intel Corporation
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

interrupt_handler() {
    exit 1
}
trap interrupt_handler SIGINT

function Info() {
    echo -e "\033[32m[Info] $@ \033[0m"
}

function Warning() {
    echo -e "\033[33;3m[Warning] $@ \033[0m"
}

function Error() {
    echo -e "\033[31m[Error] $@ \033[0m"
    exit 1
}

while [ -n "$1" ]; do
    case $1 in
    -m | --model_name)
        model_name=$2
        shift 2
        ;;
    -mp | --model_path)
        model_path=$2
        shift 2
        ;;
    -tp | --token_path)
        token_path=$2
        shift 2
        ;;
    -d | --dtype)
        case $2 in
        "bf16" | "bf16_fp16" | "bf16_int8" | "int8" | "fp16" | "bf16_int4" | "int4" | "bf16_nf4" | "nf4" | "w8a8" | "bf16_w8a8" | "w8a8_int8" | "w8a8_int4" | "w8a8_nf4")
            dtype=$2
            shift 2
            ;;
        *)
            Error "dtype must in bf16, bf16_fp16, bf16_int8, int8, fp16, bf16_int4, bf16_nf4, nf4, w8a8, bf16_w8a8, w8a8_int8, w8a8_int4, w8a8_nf4."
            exit 1
            ;;
        esac
        ;;
    -s | --sockets)
        case $2 in
        "1" | "2")
            sockets=$2
            shift 2
            ;;
        *)
            Error "sockets must in 1 or 2."
            exit 1
            ;;
        esac
        ;;
    -bs | --batch_size)
        batch_size=$2
        shift 2
        ;;
    -in | --input_tokens)
        input_tokens=$2
        shift 2
        ;;
    -out | --output_tokens)
        output_tokens=$2
        shift 2
        ;;
    -b | --beam_width)
        beam_width=$2
        shift 2
        ;;
    -i | --iter)
        iter=$2
        shift 2
        ;;
    -w | --warmup)
        warmup=$2
        shift 2
        ;;
    -c | --csv)
        csv=$2
        shift 2
        ;;
    -kvd | --kv_cache_dtype)
        kv_cache_dtype=$2
        shift 2
        ;;
    "")
        shift
        break
        ;;
    esac
done

if [ "${model_name}" == "" ]; then
    Error "Please pass a value of model name using -m or --model_name."
    exit 1
fi
if [ "${model_path}" == "" ] || [ "${token_path}" == "" ]; then
    Warning "Please pass both 'model_path' and 'token_path' at the same time if you want to use real model."
    Info "Using fake model mode now."
    export XFT_FAKE_MODEL=1
    model_path=""
    token_path=""
fi

model_path=${model_path:-"${SCRIPT_DIR}"/../examples/model_config/${model_name}/}
token_path=${token_path:-"${SCRIPT_DIR}"/../examples/model_config/${model_name}/}
dtype=${dtype:-bf16}
kv_cache_dtype=${kv_cache_dtype:-fp16}
sockets=${sockets:-1}
batch_size=${batch_size:-1}
input_tokens=${input_tokens:-32}
output_tokens=${output_tokens:-32}
beam_width=${beam_width:-1}
iter=${iter:-10}
warmup=${warmup:-2}

Info "You are using model ${model_name}, dtype ${dtype}, kvcache dtype ${kv_cache_dtype}, batch size ${batch_size}, input tokens ${input_tokens}, output tokens ${output_tokens}, beam width ${beam_width} and iteration ${iter} on ${sockets} sockets system."

Warning "The mapping method for CPU IDs in the cloud server environment is different,
        for example, (0,1), (2,3), (...) where consecutive pairs of CPU IDs belong
        to a single physical core. In this mapping relationship,
        you can enable \`export XFT_CLOUD_ENV=1\` to bind to the correct physical core."
export XFT_CLOUD_ENV=${XFT_CLOUD_ENV:-0}

benchmark_cmd="python "${SCRIPT_DIR}"/benchmark.py \
    --token_path "${token_path}" \
    --model_path "${model_path}" \
    --prompt_path "${SCRIPT_DIR}"/prompt.json \
    --model_name "${model_name}" \
    --dtype "${dtype}" \
    --kv_cache_dtype "${kv_cache_dtype}" \
    --batch_size "${batch_size}" \
    --token_in ${input_tokens}	\
    --token_out ${output_tokens} \
    --beam_width ${beam_width} \
    --iteration ${iter} \
    --warmup ${warmup}"

if [[ ${model_name} == *"llama"* ]] || [[ ${model_name} == *"baichuan-"* ]]; then
    benchmark_cmd+=" --padding=False"
fi

if [ -n $csv ]; then
    benchmark_cmd+=" --csv=$csv"
fi

if [[ ${beam_width} -eq 1 ]] && [[ ${input_tokens} -ge 1024 ]]; then
    export ENABLE_KV_TRANS=1
fi

if [[ ${input_tokens} -ge 2048 ]]; then
    export ENABLE_SKIP_MASK=1
fi

sockets_num=$(lscpu | grep "Socket(s)" | awk -F ':' '{print $2}')
cores_per_socket=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
numa_nodes=$(lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}')
# Multiply by 2 to avoid an float result in HBM flat mode that the NUMA count twice and it will be divided later.
cores_per_numa=$(($sockets_num * $cores_per_socket * 2 / $numa_nodes))

export BENCHMARK=$benchmark_cmd

if [ "${numa_nodes}" -eq 16 ]; then
    #HBM flat SNC-4 mode, Confirm that there are 8 HBM memory nodes and 8 DRAM memory nodes through "numactl -H"
    #0-7 is DRAM memory node, 8-15 is HBM node
    export OMP_NUM_THREADS=${cores_per_numa}
    Info "OMP_NUM_THREADS: ${cores_per_numa}"
    Info "SPR-HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 bash run.sh 0  8 ${OMP_NUM_THREADS} 0 : \
    -n 1 bash run.sh 1  9 ${OMP_NUM_THREADS} 1 : \
    -n 1 bash run.sh 2 10 ${OMP_NUM_THREADS} 2 : \
    -n 1 bash run.sh 3 11 ${OMP_NUM_THREADS} 3"
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 4 12 ${OMP_NUM_THREADS} 4 : \
        -n 1 bash run.sh 5 13 ${OMP_NUM_THREADS} 5 : \
        -n 1 bash run.sh 6 14 ${OMP_NUM_THREADS} 6 : \
        -n 1 bash run.sh 7 15 ${OMP_NUM_THREADS} 7"
    fi
elif [ "${numa_nodes}" -eq 8 ]; then
    #HBM SNC-4 for cache or hbm only mode
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    Info "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    Info "SPR-HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 bash run.sh 0 0 ${OMP_NUM_THREADS} 0 : \
    -n 1 bash run.sh 1 1 ${OMP_NUM_THREADS} 1 : \
    -n 1 bash run.sh 2 2 ${OMP_NUM_THREADS} 2 : \
    -n 1 bash run.sh 3 3 ${OMP_NUM_THREADS} 3"
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 4 4 ${OMP_NUM_THREADS} 4 : \
        -n 1 bash run.sh 5 5 ${OMP_NUM_THREADS} 5 : \
        -n 1 bash run.sh 6 6 ${OMP_NUM_THREADS} 6 : \
        -n 1 bash run.sh 7 7 ${OMP_NUM_THREADS} 7"
    fi
elif [[ "${numa_nodes}" -eq 4 ]] && [[ "${sockets_num}" -eq 2 ]]; then
    #HBM flat Quad-mode, Confirm that there are 2 HBM memory nodes and 2 DRAM memory nodes through "nuamctl -H"
    # or EMR SNC-2 mode
    numa_nodes_info=$(lscpu | grep "NUMA node3 CPU(s):" | awk -F ':' '{print $2}')
    if [ "$numa_nodes_info" == "" ]; then
        Info "SPR-HBM Quad mode"
        export OMP_NUM_THREADS=${cores_per_numa}
        Info "OMP_NUM_THREADS: ${cores_per_numa}"
        run_cmd="mpirun \
        -n 1 bash run.sh 0 2 ${OMP_NUM_THREADS} 0"
        if [ "$sockets" == "2" ]; then
            run_cmd+=" : \
            -n 1 bash run.sh 1 3 ${OMP_NUM_THREADS} 1"
        fi
    else
        Info "EMR SNC-2 mode"
        export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
        Info "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
        run_cmd="mpirun \
        -n 1 bash run.sh 0 0 ${OMP_NUM_THREADS} 0 : \
        -n 1 bash run.sh 1 1 ${OMP_NUM_THREADS} 1"
        if [ "$sockets" == "2" ]; then
            run_cmd+=" : \
            -n 1 bash run.sh 2 2 ${OMP_NUM_THREADS} 2 : \"
            -n 1 bash run.sh 3 3 ${OMP_NUM_THREADS} 3"
        fi
    fi
elif [[ "${numa_nodes}" -eq 4 ]] && [[ "${sockets_num}" -eq 4 ]]; then
    #4-socket spr quad mode
    Info "#SPR-SP 4-socket Quad mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    Info "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    run_cmd="mpirun \
    -n 1 bash run.sh 0 0 ${OMP_NUM_THREADS} 0"
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 1 1 ${OMP_NUM_THREADS} 1"
    fi
    if [ "$sockets" == "3" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 2 2 ${OMP_NUM_THREADS} 2"
    fi
    if [ "$sockets" == "4" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 3 3 ${OMP_NUM_THREADS} 3"
    fi
elif [ "${numa_nodes}" -eq 2 ]; then
    #SPR or hbm only or hbm cache Quad-mode or EMR non SNC-2 mode, Confirm that there are 2 DRAM memory nodes through "nuamctl -H"
    Info "Quad mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    Info "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    run_cmd="mpirun \
    -n 1 bash run.sh 0 0 ${OMP_NUM_THREADS} 0"
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 bash run.sh 1 1 ${OMP_NUM_THREADS} 1"
    fi
elif [ "${numa_nodes}" -eq 1 ]; then
    # General Test mode
    Info "General Test mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    Info "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    run_cmd="mpirun \
    -n 1 bash run.sh 0 0 ${OMP_NUM_THREADS} 0"
else
    Error "Please double check the memory nodes"
fi

echo "Run command line: ${run_cmd}"
eval "${run_cmd}"
