#!/bin/bash

# On different systems, different models and differnt modes, OMP_NUM_THREADS need to be ajusted. 
output=$(numactl -H)
first_line=$(echo "$output" | head -n 1)
nodes=$(echo "$first_line" | awk '{print $2}')

echo "memory node number: $nodes"
if [ "$nodes" -eq 16 ]; then
#HBM SNC-4 mode, Confirm that there are 8 HBM memory nodes and 8 DRAM memory nodes through "numactl -H"
#0-7 is DRAM memory node, 8-15 is HBM node
  echo "HBM SNC4 mode"
# Run Llama-7B on 1 socket HBM SNC4 mode
  OMP_NUM_THREADS=12 mpirun \
    -n 1 numactl -m 8  -N 0 sh llama-7b.sh : \
    -n 1 numactl -m 9  -N 1 sh llama-7b.sh : \
    -n 1 numactl -m 10 -N 2 sh llama-7b.sh : \
    -n 1 numactl -m 11 -N 3 sh llama-7b.sh
#    -n 1 numactl -m 12 -N 4 sh llama-7b.sh : \
#    -n 1 numactl -m 13 -N 5 sh llama-7b.sh : \
#    -n 1 numactl -m 14 -N 6 sh llama-7b.sh : \
#    -n 1 numactl -m 15 -N 7 sh llama-7b.sh

elif [ "$nodes" -eq 4 ]; then
#HBM Quad-mode, Confirm that there are 2 HBM memory nodes and 2 DRAM memory nodes through "nuamctl -H"
  echo "HBM Quad mode"
# Run Llama-7B on 1 socket HBM Quad mode
  OMP_NUM_THREADS=48 mpirun \
    -n 1 numactl -N 0  -m 2 sh llama-7b.sh : \
    -n 1 numactl -N 1  -m 3 sh llama-7b.sh

elif [ "$nodes" -eq 2 ]; then
#SPR Quad-mode, Confirm that there are 2 DRAM memory nodes through "nuamctl -H"
  echo "SPR Quad mode"
# Run Llama-7B on 1 socket SPR Quad mode
  OMP_NUM_THREADS=48 mpirun \
    -n 1 numactl -N 0  -m 0 sh llama-7b.sh

else
  echo "Please double check the memory nodes"
fi


