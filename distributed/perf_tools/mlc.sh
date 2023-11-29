#!/bin/bash
set -x

# tar -zxvf ./mlc_v3.11.tgz

cores=48
is_ali_cloud=1

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
    cpu_index=`cloud_cpu_id $cores 0`
else
    # 使用 expr 计算值并赋值给 cpu_index
    cpu_index=0-`expr $cores - 1`
fi

for ((i=1; i<=cores; i+=1)); do
    sudo ./Linux/mlc --max_bandwidth -k$cpu_index -b500m -Z
done