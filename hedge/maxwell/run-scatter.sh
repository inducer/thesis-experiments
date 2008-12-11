#! /bin/sh
rm-vis
python ~/src/experiments/hedge/maxwell/cuda-scatter.py \
  --h=1 -d cuda_dumpkernels,cuda_keep_kernels \
  --swizzle=y:-z,z:y ~/f117.ply --vis-interval=5000 "$@"
  #--swizzle=x:z,y:x,z:y ~/src/meshpy/test/ka-6d.ply --vis-interval=5000 "$@"
  #--vis-interval=100 \
  #--swizzle=x:z,y:x,z:y ~/src/meshpy/test/ka-6d.ply --vis-interval=5000 "$@"
  #--swizzle=x:z,y:x,z:y ~/yf17.ply --vis-interval=5000 "$@"
