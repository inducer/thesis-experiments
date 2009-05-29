#! /bin/sh
set -e

rm-vis
python poisson-cuda.py --cpu "$@"
mv fld.vtu fld-cpu.vtu
python poisson-cuda.py "$@"
