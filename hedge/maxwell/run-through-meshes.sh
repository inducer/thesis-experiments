#! /bin/sh
set -e
h_values=`python -c 'print " ".join(str(0.4*0.8**i) for i in range(8))'`
for o in 2 4 6; do
  for h in $h_values; do 
    python cuda-maxwell.py --order=$o --h=$h --log-file=maxwell-o$o-h$h.dat
  done
done
