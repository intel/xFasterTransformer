node=$1
nth=$2
script=$3

if [ "$node" -eq 1 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 sh $script 0 0

elif [ "$node" -eq 2 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 sh $script 0 0 : \
                  -n 1 sh $script 1 1

elif [ "$node" -eq 4 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 sh $script 0 0 : \
                  -n 1 $script 0 0 : \
                  -n 1 $script 1 1 : \
                  -n 1 $script 1 1

fi
