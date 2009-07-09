#! /bin/bash
#set -e

ROOT=`dirname "$0"`

BASE_DIR="$HOME/mpi-logs"
MCPU_FACTOR=4

for where in gpu mcpu cpu ; do 
  for loop_np in 16 12 8 4 1; do
    for o in `seq 1 9`; do 
      mesh_size=$loop_np
      np=$loop_np

      if test "$where" = "cpu"; then
        whereopt="--cpu"
      elif test "$where" = "mcpu"; then
        whereopt="--cpu"
        np=$((np*MCPU_FACTOR))
      else
        whereopt="-d cuda_plan_no_progress"
      fi

      identifier="maxwell-np$np-o$o-$where"
      complfile="$BASE_DIR/$identifier-completed"
      logfile="$BASE_DIR/$identifier.dat"
      if ! test -f "$logfile" && ! test -f "$logfile-rank0"; then
        no_logfile=true
      else
        no_logfile=false
      fi

      if (! test -f "$complfile")  || (test "$no_logfile" = "true"); then
        echo "------------------------------------------------------"
        echo "ORDER $o $where NP $np"
        echo "------------------------------------------------------"
        if ! test -f "$complfile"; then
          echo "rerunning because $complfile wasn't found"
        fi
        if test "$no_logfile" = "true"; then
          echo "rerunning because $logfile or $logfile-rank0 wasn't found"
        fi

        echo "starting in 10 seconds..."
        sleep 10 || exit
        echo "deleting leftover state files..."
        rm -f $BASE_DIR/$identifier.dat*
        rm -f "$complfile"
        echo "starting..."
        ( mpirun -hostfile ~/hostfiles/hpcgeek-gpus -np $np \
          `which python` -O $ROOT/maxwell-cuda.py --single --local-watches \
          --log-file="$logfile" --mesh-size=$mesh_size \
          --order=$o $HOPT $whereopt --steps=100 \
          --extra-features=run_target:$where,mesh_size:$mesh_size,mcpu_factor:$MCPU_FACTOR \
          "$@" && touch "$complfile")
      fi
    done
  done
done
