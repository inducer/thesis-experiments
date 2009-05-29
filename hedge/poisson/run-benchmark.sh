#! /bin/sh

set -e
rm-vis

SUM_FILE=run-summary.txt
rm -f $SUM_FILE

for o in `seq 1 9`; do
  for where in "" "--cpu"; do
    for prec in "" "--single --tol=1e-4"; do
      echo "-------------------------------------------------"
      echo "ORDER $o $where $prec"
      echo "-------------------------------------------------"
      python poisson-cuda.py --write-summary=$SUM_FILE \
        --order=$o $where $prec \
        --no-vis --no-cg-progress --max-volume=5e-5
    done
  done
done
