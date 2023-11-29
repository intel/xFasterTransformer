# xFasterTransformer

此文件夹下主要包含xFasterTransformer分布式测试的使用说明。

## prequirements
- 安装ansible依赖
```shell
cd distributed/
pip install -r requirements.txt
```

## 分布式测试

### 1. 机器环境准备
- 确保物理硬件相同, 网络在同一子网
- [Optinal] 使用NFS存放代码, 确保环境一致
- 使用ansible部署机器环境 1. 机器两两之间的免密ssh; 2. /etc/hosts文件下的 ip<->hosts映射;
- [Optinal] 使用perf_tool的工具抓取一下环境性能:
      1. svr_info
      2. mlc 内存带宽
      3. IMB-MPI1(all_reduce)看oneCCL的连通性和性能
      ...

### 2. XFT环境准备
- 使用魔法下载XFT源码
```shell
git clone -b test/distributed https://github.com/intel/xFasterTransformer.git
```
- oneCCL环境配置
```shell
cd 3rdparty
sh prepare_oneccl.sh
source ./oneCCL/build/_install/env/setvars.sh

# 测试oneCCL的连通性, IP_A,IP_B替换为机器IP
cd ./oneCCL/build
mpirun -print-rank-map -prot -n 2 -ppn 1 -hosts IP_A,IP_B ./_install/examples/benchmark/benchmark
```
- 编译XFT
```shell
mkdir build && cd build
cmake ..
http_proxy="添加魔法" https_proxy="添加魔法" make -j
```

### 3. XFT分布式测试
修改run_benchmark.sh代码.