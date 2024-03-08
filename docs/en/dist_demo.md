EMR is recently being gradually launched on Alibaba Cloud. In the EMR launch media event, Intel demonstrated the live demo of the distributed Qwen-72B, achieving excellent on-site results. This BKM will use the same approach, EMR*4 + eRDMA network, to guide everyone in building a distributed inference demo of Qwen-72B using XFT ([Intel/xFasterTransformer](https://github.com/intel/xFasterTransformer)).

[XFT](https://github.com/intel/xFasterTransformer)(xFasterTransformer) is an exceptionally optimized solution for large language models (LLM) on the X86 platform, similar to FasterTransformer on the GPU platform. xFasterTransformer can operate in distributed mode across multiple sockets and nodes to support inference on larger models. Additionally, it provides both C++ and Python APIs, spanning from high-level to low-level interfaces, making it easy to adopt and integrate.

[eRDMA](https://help.aliyun.com/zh/ecs/user-guide/erdma-overview) (Elastic Remote Direct Memory Access) is Alibaba Cloud's self-developed elastic RDMA network in the cloud. The underlying link reuses the VPC network and employs a self-developed full-stack Congestion Control (CC) algorithm. While enjoying the high throughput and low latency characteristics of traditional RDMA networks, eRDMA can also support large-scale RDMA networking with sub-second latency. It is compatible with traditional HPC applications as well as traditional TCP/IP applications.

Based on eRDMA, you can deploy HPC application software in the cloud to obtain a cost-effective, more elastic high-performance application cluster. Alternatively, you can replace the VPC network with an eRDMA network to accelerate the performance of your other applications.
## Hardware Configuration
Purchase cloud instances of the following configuration types on Alibaba Cloud:
### Instance 1 & 2 & 3 & 4
name | xftest01
-- | --
user | root
passwd | xxxxxx
OS | Alibaba Cloud Linux 3.2104 LTS 64位
CPU Model | INTEL(R) XEON(R) PLATINUM 8575C
NUMA node0 CPU(s): | 0-95
NIC0 (with eRDMA Capability) | eth0 (192.168.0.1/2/3/4)
disk | 200GB
NFS-disk(mount in /mnt) | 1TB
### Topology of Rank
```shell

    ID |               Node name |             IP | Ranks
     0 |          EMR instance 0 |    192.168.0.1 | 0
     1 |          EMR instance 1 |    192.168.0.2 | 1
     2 |          EMR instance 2 |    192.168.0.3 | 2
     3 |          EMR instance 3 |    192.168.0.4 | 3

    ID |           0            1            2            3 
     0 |         shm        eRDMA        eRDMA        eRDMA 
     1 |       eRDMA          shm        eRDMA        eRDMA 
     2 |       eRDMA        eRDMA          shm        eRDMA 
     3 |       eRDMA        eRDMA        eRDMA          shm 
```
**Important Notes:**

**1. Ensure configuration deployment sets**([结合部属集策略实现更低的eRDMA时延](https://developer.aliyun.com/article/1385886)).
Alibaba Cloud ECS provides deployment set strategies that control the physical distribution of ECS instances. Deployment sets support various strategies:
   - **High Availability Strategy**: All ECS instances in the deployment set are strictly distributed across different physical servers within a specified region, ensuring high availability of business on ECS instances and the underlying physical server's disaster recovery capability.
   - **Low Latency Strategy**: In this mode, all ECS instances in the deployment set are deployed as centrally as possible within the same network topology range in the available zone, reducing network communication latency.

   We know that RDMA itself has the characteristics of low latency and high throughput. In practical use, it is also influenced by the actual physical network distance: the farther the distance, the greater the latency between nodes. In Alibaba Cloud, we can combine deployment set strategies to enable ECS to provide elastic RDMA acceleration, obtaining lower latency as much as possible.

**2. Ensure optimal configuration for eRDMA:**
   - **Disable Delay ACK**: Currently, Alibaba support is needed for manual modification of Alibaba Cloud network parameters. Waiting for the new version update, eadm tool can be used for 'Delay ACK' configuration by users themselves.
   - **Enable Congestion Control (CC) algorithm**: running `$ eadm conf -d erdma_0 -t cc -v 1`

**3. Ensure the following VMs are distributed on different physical resources.**
   Cloud instances do not have isolated memory bandwidth resources. If deployed on the same physical resource, there may be bandwidth contention issues.

**4. Ensure consistent code environment; machines mount files to /mnt directory using NFS for file synchronization.**

## Software Configuration
### Basic Environment
```shell
# Initially, we need to perform some basic environment configuration, such as:
# 1. Install system dependencies;
# 2. Install Python environment dependencies;
# 3. Configure passwordless login between machines;
# ...

$ yum install -y htop tmux git git-lfs python38 python38-pip numactl numactl-devel"

$ pip3.8 install --upgrade pip

$ cd /mnt/xFasterTransformer/
# install torch / transformers dependencies
$ pip3.8 install -r requirements.txt

# install distributed test dependencies.
#  (optinal) install jpeg dependencies:
#  $ yum -y install libjpeg-turbo-devel
$ cd distributed/
$ pip3.8 install -r requirements.txt

# modify host configuration
# Default user is using root. you can modify from ansible.cfg.
$ vim hosts
  [all_hosts:vars]
  ansible_ssh_pass=<server password> 
  
  # modify host IP address
  [all_hosts]
  192.168.0.1
  192.168.0.2
  192.168.0.3
  192.168.0.4

# replace the default python + pip version
$ ansible all_hosts -m shell -a "rm -rf /usr/bin/pip /usr/bin/python && ln -s /usr/bin/pip3.8 /usr/bin/pip && ln -s /usr/bin/python3.8 /usr/bin/python"

# using the ansible ping plugin to check the network.
$ ansible all_hosts -m ping
...
192.168.0.1 | SUCCESS => {
    "ansible_facts": {
        "discovered_interpreter_python": "/usr/bin/python3.6"
    },
    "changed": false,
    "ping": "pong"
}
...

# Configure passwordless login between machines
$ ansible-playbook 000-ssh-wo-passwd-update-hosts.xml

```
### oneCCL Configuration
```shell
# Download the deployment environment on NFS and
# synchronize code files across multiple machines.
$ cd /mnt
$ git clone -b test/distributed https://github.com/intel/xFasterTransformer.git

$ cd /mnt/xFasterTransformer/

# Compile and install the oneCCL environment.
$ cd 3rdparty/
$ sh prepare_oneccl.sh

# Synchronize the multi-node environment configuration.
$ cd /mnt/xFasterTransformer/distributed/
$ ansible all_hosts -m shell -a "echo 'source /mnt/xFasterTransformer/3rdparty/oneccl/build/_install/env/setvars.sh' >> ~/.bashrc"

# Update environment variables, run benchmarks,
# and if the output is similar to the reference
# content below, then the oneCCL configuration is successful.
$ bash
$ cd /mnt/xFasterTransformer/3rdparty/oneccl/build/
$ mpirun -print-rank-map -prot -n 4 -ppn 1 -hosts 192.168.0.1,192.168.0.2 ./_install/examples/benchmark/benchmark
(192.168.0.1:0)
(192.168.0.2:1)

    ID |               Node name |             IP | Ranks
     0 | iZ2ze2krbyeyuro5gcr1uiZ | 192.168.0.1 | 0
     1 | iZ2zeb3qtvacqss8re3pcrZ | 192.168.0.2 | 1
    ID |             0             1 
     0 | verbs;ofi_rxm verbs;ofi_rxm 
     1 | verbs;ofi_rxm verbs;ofi_rxm 


options:
  processes:      2
  backend:        host
  iters:          16
  warmup_iters:   16
  iter_policy:    auto
  buf_count:      1
  min_elem_count: 1
  max_elem_count: 128
  elem_counts:    [1 2 4 8 16 32 64 128]
  check:          last
  cache:          1
  inplace:        0
  collectives:    allreduce 
  datatypes:      float32 
  reductions:     sum 
  extended info:  auto
  csv_filepath:   

datatype: float32
reduction: sum

#------------------------------------------------------------
# Benchmarking: allreduce 
# #processes: 2
#------------------------------------------------------------

        #bytes  #repetitions   t_min[usec]   t_max[usec]   t_avg[usec]  stddev[%]
             4            16         30.50         30.86         30.68       0.59
             8            16         29.70         29.77         29.73       0.11
            16            16         29.81         30.08         29.95       0.44
            32            16         30.25         30.98         30.62       1.20
            64            16         30.41         31.09         30.75       1.12
           128            16         30.11         31.14         30.62       1.68
           256            16         30.47         30.52         30.49       0.08
           512            16         30.28         30.41         30.34       0.21

# All done

```
## XFT setup
```shell
$ cd /mnt/xFasterTransformer/
$ pip install -r requirements.txt

$ mkdir build && cd build

# Compile and install; if there are any issues, feel free to submit an issue.
# https://github.com/intel/xFasterTransformer/issues/new
# cmake -DPython_EXECUTABLE=/usr/bin/python3.8 ..
$ cmake .. && make -j
```

## Demo setup

### 1. Qwen 模型下载

可以从下面两个链接下载, 模型文件放到 /mnt/model 下
- https://huggingface.co/Qwen/Qwen-72B-Chat
- https://modelscope.cn/models/qwen/Qwen-72B-Chat/summary

### 2. 模型转换
使用如下命令会将开源模型转化为XFT可运行的模型格式:
``` shell
$ cd tools && python qwen_convert.py -i /mnt/model/Qwen-72B-Chat -o /mnt/model/Qwen-72B-Chat-xft
```
稍等片刻, 转好的模型文件会在 /mnt/model/Qwen-72B-Chat-xft 目录下.

### 3. 模型部署

```
# 安装demo依赖
cd /mnt/xFasterTransformer/examples/web_demo && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 单机版本
python Qwen.py -t /mnt/model/Qwen-72B-Chat -m /mnt/model/Qwen-72B-Chat-xft -d bf16

# 分布式版本需要使用mpirun
mpirun -iface=${IFACE} $MPI_DEBUG \
    -n 1 -hosts ${IP_A} numactl -C `绑定具体的物理核` -m 0 python Qwen.py -t /mnt/data/Qwen-72B-Chat -m /mnt/data/Qwen-72B-Chat-xft --dtype=bf16 : \
    -n 1 -hosts ${IP_B} numactl -C `绑定具体的物理核` -m 0 python Qwen.py -t /mnt/data/Qwen-72B-Chat -m /mnt/data/Qwen-72B-Chat-xft --dtype=bf16 : \
    -n 1 -hosts ${IP_C} numactl -C `绑定具体的物理核` -m 0 python Qwen.py -t /mnt/data/Qwen-72B-Chat -m /mnt/data/Qwen-72B-Chat-xft --dtype=bf16 : \
    -n 1 -hosts ${IP_D} numactl -C `绑定具体的物理核` -m 0 python Qwen.py -t /mnt/data/Qwen-72B-Chat -m /mnt/data/Qwen-72B-Chat-xft --dtype=bf16
```
分布式运行可以参考如下脚本.
```
# run.sh

#!/bin/bash

set -x

interrupt_handler() {
  exit 1
}
trap interrupt_handler SIGINT

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

############# PATH configuration #############
current_dir=$(pwd)
workspace_dir=$(echo $current_dir | sed 's|\(.*\/xFasterTransformer\).*|\1|')
build_dir=$(echo $workspace_dir/build)

# change the workspace status
if [ ! -d $build_dir ]; then
    echo "[Error] please build project in $build_dir"
    exit 1
fi

logs_dir=$(echo $current_dir/logs_erdma/`date "+%Y-%m-%d-%H-%M-%S"`)
mkdir -p $logs_dir

############# HW configuration #############
IFACE=eth1
IP_A=172.31.0.104
IP_B=172.31.0.106
IP_C=172.31.0.100
IP_D=172.31.0.102

# enable it if testing at a cloud environment
export XFT_CLOUD_ENV=1

# sync manual
# scp -r $workspace_dir/* $IP_B:$workspace_dir/

# set OpenMP lib.
export LD_PRELOAD=$workspace_dir/3rdparty/mklml/lib/libiomp5.so

# enable log https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-11/other-environment-variables.html#GUID-8357A7B3-5494-48AF-AA32-CAA4A778D195
# I_MPI_DEBUG=1
# FI_LOG_LEVEL=debug

# enable TCP
# export FI_TCP_IFACE=eth0
# export I_MPI_OFI_PROVIDER="tcp;ofi_rxm"

# enable eRDMA
export FI_VERBS_IFACE=eth1
export FI_PROVIDER="verbs;ofi_rxm"
export FI_OFI_RXM_USE_SRX=0
export FI_VERBS_RX_IOV_LIMIT=1

# export FI_OFI_RXM_BUFFER_SIZE=32768

############# OneCCL configuration #############
# export CCL_ALLREDUCE=recursive_doubling
export CCL_ALLREDUCE="recursive_doubling:0-16384;2d:16385-524288;nreduce:524289-max"

export CCL_PROCESS_LAUNCHER=none

export CCL_WORKER_COUNT=1

#for 48 core * 2
#set CCL_WORKER_AFFINITY if necessary
export CCL_WORKER_AFFINITY=95

############# XFT configuration #############
BENCHMARK=$build_dir/example
export XFT_ONECCL=1
#export XFT_ONECCL_BF16=1
export XFT_COMM_TIME=0
export XFT_FAKE_MODEL=0
export XFT_TIMELINE=0

# open for MPI debug information
# MPI_DEBUG="-prot -verbose -print-rank-map -print-all-exitcodes -outfile-pattern=run_output_std.log -errfile-pattern=run_output_err.log"

############# BENCHMARK configuration #############
export OMP_NUM_THREADS=48
export LD_LIBRARY_PATH=$workspace_dir/build:$LD_LIBRARY_PATH

mpirun -iface=${IFACE} $MPI_DEBUG \
    -n 1 -hosts ${IP_A} numactl -C `cloud_cpu_id 48 0` -m 0 python Qwen.py --token_path=/mnt/data/Qwen-72B-Chat --model_path=/mnt/data/Qwen-72B-Chat-xft --dtype=bf16_fp16 : \
    -n 1 -hosts ${IP_B} numactl -C `cloud_cpu_id 48 0` -m 0 python Qwen.py --token_path=/mnt/data/Qwen-72B-Chat --model_path=/mnt/data/Qwen-72B-Chat-xft --dtype=bf16_fp16 : \
    -n 1 -hosts ${IP_C} numactl -C `cloud_cpu_id 48 0` -m 0 python Qwen.py --token_path=/mnt/data/Qwen-72B-Chat --model_path=/mnt/data/Qwen-72B-Chat-xft --dtype=bf16_fp16 : \
    -n 1 -hosts ${IP_D} numactl -C `cloud_cpu_id 48 0` -m 0 python Qwen.py --token_path=/mnt/data/Qwen-72B-Chat --model_path=/mnt/data/Qwen-72B-Chat-xft --dtype=bf16_fp16

# numactl -C `cloud_cpu_id 48 0` -m 0 python Qwen.py --token_path=/mnt/data/Qwen-72B-Chat --model_path=/mnt/data/Qwen-72B-Chat-xft --dtype=bf16_fp16
```

### Qwen 72B distributed benchmark
```shell
$ cd /mnt/xFasterTransformer/distributed/
# 修改分布式脚本:
#   1. ############# HW configuration #############
#        a. 修改IFACE为指定网卡;
#        b. 修改IP_A/IP_B/IP_C/...;
#        c. 如果在云上测试则: export XFT_CLOUD_ENV=1;
#        d. TCP or eRDMA 二选一: 打开# enable TCP/eRDMA 注释;
#   2. ############# XFT configuration #############
#        a. XFT_COMM_TIME=1: 打印单次allreduce时间日志;
#        a. XFT_FAKE_MODEL=1: 模型文件未就位可以使用fake model, 
#                             此时模型输出没有意义, 仅仅性能测试使用;
$ vim run_benchmark.sh
$ bash run_benchmark.sh
```

## 参考链接
- [结合部属集策略实现更低的eRDMA时延](https://developer.aliyun.com/article/1385886)
- [TCP-IP详解：Delay ACK](https://blog.csdn.net/wdscq1234/article/details/52430382)
- [弹性RDMA的技术解析与实践](https://developer.aliyun.com/article/1286492)
- [阿里云eRDMA简要解读，用云计算推动RDMA通用化和平民化](http://www.dostor.com/p/79075.html)
- [eRDMA概述](https://help.aliyun.com/zh/ecs/user-guide/erdma-overview)