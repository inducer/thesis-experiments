#! /bin/sh
rm -f output.dat 
mpirun -n 56 -hostfile ~/hostfiles/hpcgeek-all `which python` mrab-stability.py
