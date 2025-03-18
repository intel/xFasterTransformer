#!/bin/bash
# set -x

# $1: first token numa node.
# $2: second token numa node.
# $3: thread nums.
# $4: thread nums amount.

function cloud_cpu_id() {
	num_threads=$1
	num_iters=$2
	start_index=$(($num_threads * 2 * $num_iters))
	iterations=$(($start_index + $num_threads * 2))

	cpu_index="$start_index"

	for ((i = start_index + 2; i < iterations; i += 2)); do
		cpu_index+=",$i"
	done
	echo $cpu_index
}

cores_per_socket=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
numa_nodes=$(lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}')
remainder=$(($cores_per_socket % $3))
front_increment=0
back_increment=0

if [ $remainder -gt 0 ]; then
	index_offset=0
	if [ $(expr $3 \* $4 + $3) -gt $cores_per_socket ]; then
		index_offset=$(($numa_nodes / 2))
	fi

	if [ $remainder -lt $(expr $4 - $index_offset) ]; then
		front_increment=$remainder
	else
		front_increment=$(expr $4 - $index_offset)
	fi

	if [ $remainder -lt $(expr $4 + 1 - $index_offset) ]; then
		back_increment=$remainder
	else
		back_increment=$(expr $4 + 1 - $index_offset)
	fi

	if [ $index_offset -gt 0 ]; then
		front_increment=$(expr $front_increment + $remainder)
		back_increment=$(expr $back_increment + $remainder)
	fi

fi

if [ "$XFT_CLOUD_ENV" -eq 1 ]; then
	# 调用 get_value 函数并将结果赋值给 cpu_index
	cpu_index=$(cloud_cpu_id $3 $4)
else
	# 使用 expr 计算值并赋值给 cpu_index
	cpu_index=$(expr $3 \* $4 + $front_increment)-$(expr $3 \* $4 + $3 - 1 + $back_increment)
fi

# echo FIRST_TOKEN_WEIGHT_LOCATION=$1 NEXT_TOKEN_WEIGHT_LOCATION=$2 OMP_NUM_THREADS=$3 \
# 	numactl --all -C $cpu_index -p $2 $BENCHMARK

FIRST_TOKEN_WEIGHT_LOCATION=$1 NEXT_TOKEN_WEIGHT_LOCATION=$2 OMP_NUM_THREADS=$3 \
	numactl --all -C $cpu_index -p $2 $BENCHMARK
