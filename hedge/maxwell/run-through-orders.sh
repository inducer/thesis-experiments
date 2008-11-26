#! /bin/sh
set -e
for o in `seq 1 9`; do 
  if test $o -ge 8; then
    h=0.10
  else
    h=0.075
  fi
  python cuda-maxwell.py --order=$o --h=$h "$@"
done
