#! /bin/sh
rm-vis
python ~/src/experiments/hedge/maxwell/cuda-scatter.py \
  --h=5 -d cuda_dumpkernels,cuda_keep_kernels --vis-interval=5000 \
  --swizzle=y:z,z:y ~/f117.ply "$@"
  #--swizzle=x:z,y:x,z:y ~/src/meshpy/test/ka-6d.ply "$@"
  #--swizzle=x:z,y:x,z:y ~/yf17.ply "$@"
