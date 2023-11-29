# xFasterTransformer

workspace performance test
- 系统配置: svr info
- memory bandwidth: mlc
- 网络测试: IMB-MPI1

## prequirements
- 安装ansible依赖
```shell
pip install -r requirements.txt
```
## memory bandwidth
### download mlc
https://www.intel.com/content/www/us/en/download/736633/intel-memory-latency-checker-intel-mlc.html
```shell
$ sudo ./mlc --max_bandwidth -b500m -Z

$ sudo ./mlc --max_bandwidth -k0 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,8 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,8,10 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,8,10,12 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,8,10,12,14 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,8,10,12,14,16 -b500m -Z
$ sudo ./mlc --max_bandwidth -k0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94 -b500m -Z
```
## network benchmark

```shell
[root@xftest001 distributed]# ./IMB-MPI1 allreduce
#----------------------------------------------------------------
#    Intel(R) MPI Benchmarks 2021.3, MPI-1 part
#----------------------------------------------------------------
# Date                  : Mon Nov 20 19:56:00 2023
# Machine               : x86_64
# System                : Linux
# Release               : 5.10.134-15.al8.x86_64
# Version               : #1 SMP Thu Jul 20 00:44:04 CST 2023
# MPI Version           : 3.1
# MPI Thread Environment: 


# Calling sequence was: 

# ./IMB-MPI1 allreduce 

# Minimum message length in bytes:   0
# Maximum message length in bytes:   4194304
#
# MPI_Datatype                   :   MPI_BYTE 
# MPI_Datatype for reductions    :   MPI_FLOAT 
# MPI_Op                         :   MPI_SUM  
# 
# 

# List of Benchmarks to run:

# Allreduce

#----------------------------------------------------------------
# Benchmarking Allreduce 
# #processes = 1 
#----------------------------------------------------------------
       #bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]
            0         1000         0.04         0.04         0.04
            4         1000         0.04         0.04         0.04
            8         1000         0.04         0.04         0.04
           16         1000         0.04         0.04         0.04
           32         1000         0.04         0.04         0.04
           64         1000         0.04         0.04         0.04
          128         1000         0.04         0.04         0.04
          256         1000         0.04         0.04         0.04
          512         1000         0.05         0.05         0.05
         1024         1000         0.05         0.05         0.05
         2048         1000         0.05         0.05         0.05
         4096         1000         0.07         0.07         0.07
         8192         1000         0.08         0.08         0.08
        16384         1000         0.13         0.13         0.13
        32768         1000         0.75         0.75         0.75
        65536          640         1.49         1.49         1.49
       131072          320         3.05         3.05         3.05
       262144          160         6.17         6.17         6.17
       524288           80        12.11        12.11        12.11
      1048576           40        25.30        25.30        25.30
      2097152           20       146.44       146.44       146.44
      4194304           10       291.86       291.86       291.86


# All processes entering MPI_Finalize
```

## 分布式测试