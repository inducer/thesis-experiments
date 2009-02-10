import pycuda.autoinit
import pycuda.driver as drv
import numpy

mod = drv.SourceModule("""
__shared__ unsigned x;
__global__ void atomic_test(float *dest)
{
  __syncthreads();
  x = 0;
  __syncthreads();
  atomicAdd(&x, 1);
  __syncthreads();
  dest[threadIdx.x] = x;
}
""")

multiply_them = mod.get_function("atomic_test")

dest = numpy.empty((400,), dtype=numpy.float32)
multiply_them(drv.Out(dest), block=(400,1,1))

print dest

