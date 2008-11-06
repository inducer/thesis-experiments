#! /bin/sh
set -e

for o in `seq 1 9`; do 
  for df in \
    "" \
    cuda_no_smem_matrix \
    cuda_no_microblock \
    cuda_no_smem_matrix,cuda_no_microblock; do
    if test $o -ge 8; then
      h=0.10
    else
      h=0.075
    fi
    echo "o=$o h=$h df=$df"
    python cuda-maxwell.py \
      --order=$o \
      --h=$h \
      --log-file="maxwell-$df-%(order)s.dat" \
      --debug-flags="$df"
  done
done
