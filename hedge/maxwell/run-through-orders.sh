#! /bin/sh
set -e

ROOT=`dirname "$0"`

if test "$1" = "--adapt-h"; then
  ADAPT_H=1
  shift
fi
for o in `seq 1 9`; do 
  if test "$ADAPT_H" != ""; then
    if test $o -ge 8; then
      HOPT="--h=0.10"
    else
      HOPT="--h=0.075"
    fi
  fi
  python $ROOT/maxwell-cuda.py --order=$o $HOPT "$@"
done
