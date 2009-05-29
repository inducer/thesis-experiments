#! /bin/bash
#set -e

ROOT=`dirname "$0"`

for where in gpu cpu; do 
  for np in 16 12 8 4 1; do
    for o in `seq 1 9`; do 
      if test "$where" = "cpu"; then
        whereopt="--cpu"
      else
        whereopt="-d cuda_plan_no_progress"
      fi

      echo "------------------------------------------------------"
      echo "ORDER $o $where NP $np"
      echo "------------------------------------------------------"
      mpirun -hostfile ~/hostfiles/hpcgeek-gpus -np $np \
        `which python` $ROOT/maxwell-cuda.py --single --local-watches \
        --log-file="maxwell-np$np-o$o-$where.dat" --mesh-size=$((np)) \
        --order=$o $HOPT $whereopt --steps=100 \
        "$@"
    done
  done
done
