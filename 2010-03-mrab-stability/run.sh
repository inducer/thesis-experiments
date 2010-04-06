#! /bin/sh
rm -f output*.dat 
mpirun -n 48 -hostfile ~/hostfiles/hpcgeek-all `which python` mrab_stability.py
