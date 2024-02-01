#!/bin/bash
set -x

# $1: first token numa node.
# $2: second token numa node.
# $3: thread nums.
# $4: thread nums amount.

function cloud_cpu_id() {
	num_threads=$1
	num_iters=$2
	start_index=$(($num_threads*2*$num_iters))
	iterations=$(($start_index+$num_threads*2))

	cpu_index="$start_index"

	for ((i=start_index+2; i<iterations; i+=2)); do
		cpu_index+=",$i"
	done
	echo $cpu_index
}

if [ "$is_ali_cloud" -eq 1 ]; then
    # 调用 get_value 函数并将结果赋值给 cpu_index
    cpu_index=`cloud_cpu_id $3 $4`
else
    # 使用 expr 计算值并赋值给 cpu_index
    cpu_index=`expr $3 \* $4`-`expr $3 \* $4 + $3 - 1`
fi

FIRST_TOKEN_WEIGHT_LOCATION=$1 NEXT_TOKEN_WEIGHT_LOCATION=$2 OMP_NUM_THREADS=$3 \
	numactl --all -C $cpu_index -m $2 $BENCHMARK