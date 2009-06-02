#! /bin/sh
set -e

ROOT=`dirname "$0"`

if test "$1" = "--adapt-h"; then
  ADAPT_H=1
  shift
  if test "$1" == "--single"; then
    echo "h adaptation (single) turned on"
    COARSE_H=0.10
    FINE_H=0.075
    SWITCH_ORDER=8
  else
    echo "h adaptation (double) turned on"
    COARSE_H=0.12
    FINE_H=0.085
    SWITCH_ORDER=7
  fi
fi

for o in `seq 1 9`; do 
  if test "$ADAPT_H" != ""; then
    if test $o -ge $SWITCH_ORDER; then
      HOPT="--h=$COARSE_H"
    else
      HOPT="--h=$FINE_H"
    fi
  fi
  python -O $ROOT/maxwell-cuda.py --order=$o $HOPT "$@"
done
