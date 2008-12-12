#! /bin/sh
set -e

h_values=`python -c 'print " ".join(str(0.3*0.9**i) for i in range(4))'`
for o in `seq 1 9`; do 
  for h in $h_values; do 
      python cuda-maxwell.py \
        --order=$o \
        --h=$h \
        --log-file="maxwell-conv-$h-$o.dat" -d cuda_no_plan
  done
done

