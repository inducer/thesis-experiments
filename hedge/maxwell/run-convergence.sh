#! /bin/sh
set -e

h_values=`python -c 'print " ".join(str(0.3*0.85**i) for i in range(4))'`
for h in $h_values; do 
  for o in `seq 1 5`; do 
      python cuda-maxwell.py \
        --order=$o \
        --h=$h \
        --log-file="maxwell-conv-$h-$o.dat" -d cuda_no_plan
  done
  for o in `seq 6 9`; do 
      python cuda-maxwell.py \
        --order=$o \
        --h=$h \
        --log-file="maxwell-conv-$h-$o.dat" 
  done
done

