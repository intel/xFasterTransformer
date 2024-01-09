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

WEIGHT_LOCATION() {
    if [ "$#" -ne 2 ]; then
        echo "error: must get two input." >&2
        return 1
    fi

    local result="-env FIRST_TOKEN_WEIGHT_LOCATION $1 -env NEXT_TOKEN_WEIGHT_LOCATION $2"
    echo "$result"
}

while [ -n "$1" ]  
do	
	case $1 in
        -m | --model_name)
		model_name=$2
		shift 2
        ;;  
        -d | --dtype)
		case $2 in
            "bf16" | "bf16_fp16" | "bf16_int8" | "int8" | "fp16" | "bf16_int4" | "int4" | "bf16_nf4" | "nf4" | "w8a8" | "bf16_w8a8" | "w8a8_int8" | "w8a8_int4" | "w8a8_nf4")
            dtype=$2
            shift 2
            ;;
            *)
            echo "dtype must in bf16, bf16_fp16, bf16_int8, int8, fp16, bf16_int4, bf16_nf4, nf4, w8a8, bf16_w8a8, w8a8_int8, w8a8_int4, w8a8_nf4."
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
            echo "sockets must in 1 or 2."
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
        "")
        shift
        break
        ;;
    esac 
done

if [ "${model_name}" == "" ]; then
    echo "Please pass a value of model name using -m or --model_name."
    exit 1
fi
dtype=${dtype:-bf16}
sockets=${sockets:-1}
batch_size=${batch_size:-1}
input_tokens=${input_tokens:-32}
output_tokens=${output_tokens:-32}
beam_width=${beam_width:-1}
iter=${iter:-10}

echo "You are using model ${model_name}, dtype ${dtype}, batch size ${batch_size}, input tokens ${input_tokens}, output tokens ${output_tokens}, beam width ${beam_width} and iteration ${iter} on ${sockets} sockets system."

# Example here is using fake model, you can use real model as well
export XFT_FAKE_MODEL=1
if [[ ${model_name} == *"chatglm3"* ]]; then
    model_path="${SCRIPT_DIR}"/../examples/model_config/chatglm2-6b/
else
    model_path="${SCRIPT_DIR}"/../examples/model_config/${model_name}/
fi

benchmark_cmd="python "${SCRIPT_DIR}"/benchmark.py \
    --token_path "${model_path}" \
    --model_path "${model_path}" \
    --prompt_path "${SCRIPT_DIR}"/prompt.json \
    --model_name "${model_name}" \
    --dtype "${dtype}" \
    --batch_size "${batch_size}" \
    --token_in ${input_tokens}	\
    --token_out ${output_tokens} \
    --beam_width ${beam_width} \
    --iteration ${iter}"

if [[ ${model_name} == *"llama"* ]] || [[ ${model_name} == *"baichuan-"* ]]; then
    benchmark_cmd+=" --padding=False"
fi

sockets_num=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
cores_per_socket=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
numa_nodes=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}'`
# Multiply by 2 to avoid an float result in HBM flat mode that the NUMA count twice and it will be divided later.
cores_per_numa=$(( $sockets_num * $cores_per_socket * 2 / $numa_nodes ))

if [ "${numa_nodes}" -eq 16 ]; then
    #HBM flat SNC-4 mode, Confirm that there are 8 HBM memory nodes and 8 DRAM memory nodes through "numactl -H"
    #0-7 is DRAM memory node, 8-15 is HBM node
    export OMP_NUM_THREADS=${cores_per_numa}
    echo "OMP_NUM_THREADS: ${cores_per_numa}"
    echo "HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 $(WEIGHT_LOCATION 0  8) numactl -p 8  -N 0 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 1  9) numactl -p 9  -N 1 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 2 10) numactl -p 10 -N 2 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 3 11) numactl -p 11 -N 3 ${benchmark_cmd} "
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 $(WEIGHT_LOCATION 4 12) numactl -p 12 -N 4 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 5 13) numactl -p 13 -N 5 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 6 14) numactl -p 14 -N 6 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 7 15) numactl -p 15 -N 7 ${benchmark_cmd} "
    fi
elif [ "${numa_nodes}" -eq 8 ]; then
    #HBM SNC-4 for cache or hbm only mode
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    echo "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    echo "HBM SNC4 mode"
    run_cmd="mpirun \
    -n 1 $(WEIGHT_LOCATION 0 0) numactl -m 0 -N 0 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 1 1) numactl -m 1 -N 1 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 2 2) numactl -m 2 -N 2 ${benchmark_cmd} : \
    -n 1 $(WEIGHT_LOCATION 3 3) numactl -m 3 -N 3 ${benchmark_cmd} "
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 $(WEIGHT_LOCATION 4 4) numactl -m 4 -N 4 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 5 5) numactl -m 5 -N 5 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 6 6) numactl -m 6 -N 6 ${benchmark_cmd} : \
        -n 1 $(WEIGHT_LOCATION 7 7) numactl -m 7 -N 7 ${benchmark_cmd} "
    fi
elif [ "${numa_nodes}" -eq 4 ]; then
    #HBM flat Quad-mode, Confirm that there are 2 HBM memory nodes and 2 DRAM memory nodes through "nuamctl -H"
    echo "HBM Quad mode"
    export OMP_NUM_THREADS=${cores_per_numa}
    echo "OMP_NUM_THREADS: ${cores_per_numa}"
    run_cmd="mpirun \
    -n 1 $(WEIGHT_LOCATION 0 2) numactl -p 2 -N 0 ${benchmark_cmd} "
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 $(WEIGHT_LOCATION 1 3) numactl -p 3 -N 1 ${benchmark_cmd} "
    fi
elif [ "${numa_nodes}" -eq 2 ]; then
    #SPR or hbm only or hbm cache Quad-mode, Confirm that there are 2 DRAM memory nodes through "nuamctl -H"
    echo "SPR Quad mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    echo "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    run_cmd="mpirun \
    -n 1 $(WEIGHT_LOCATION 0 0) numactl -N 0 -m 0 ${benchmark_cmd}"
    if [ "$sockets" == "2" ]; then
        run_cmd+=" : \
        -n 1 $(WEIGHT_LOCATION 1 1) numactl -m 1 -N 1 ${benchmark_cmd} "
    fi
else
    echo "Please double check the memory nodes"
fi

echo "Run command line: ${run_cmd}"
eval "${run_cmd}"
