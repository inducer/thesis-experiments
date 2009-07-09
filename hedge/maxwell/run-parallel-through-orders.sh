#! /bin/bash
#set -e

ROOT=`dirname "$0"`

BASE_DIR="$HOME/mpi-logs"

for where in gpu cpu; do 
  for np in 16 12 8 4 1; do
    for o in `seq 1 9`; do 
      if test "$where" = "cpu"; then
        whereopt="--cpu"
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
          --log-file="$logfile" --mesh-size=$((np)) \
          --order=$o $HOPT $whereopt --steps=100 \
          "$@" && touch "$complfile")
      fi
    done
  done
done
