#include <stdio.h>
#include <stdlib.h>

// defines ----------------------------------------------------------------

#define THREADS_PER_BLOCK 384

__shared__ unsigned oop;
__shared__ unsigned bep;

extern "C" __global__ void sift()
{
  if (threadIdx.x == 0)
  {
    bep = 10000;
  }
  __syncthreads();
  
  unsigned n_pt = threadIdx.x;

  unsigned blah, blubb;
  for (blah = 0; blah < 20; ++blah)
  {
    // sift ---------------------------------------------------------------
    
    for (blubb = 0; blubb < 20; ++blubb)
    {
      if (n_pt < bep)
        n_pt += THREADS_PER_BLOCK;
      else
        break;

    }

    if (threadIdx.x==0)
      atomicAnd(&oop, 0);
    __syncthreads();
    if (n_pt >= bep)
      atomicAdd(&oop, 1);
    __syncthreads();
    if (oop == THREADS_PER_BLOCK)
      break;
  }
}

int main(int argc, char **argv)
{
  int dev = 0;
  if (argc == 2)
    dev = atoi(argv[1]);

  printf("using device %d\n", dev);
  cudaSetDevice(dev);

  puts("V72\n");
  dim3 grid(100,100);
  dim3 block(384,1);
  for (int i = 0; i < 40; ++i)
  {
    printf("loop %d\n", i);
    sift<<<grid,block>>>();
    cudaError e = cudaThreadSynchronize();
    if (e)
      printf("ret: %d\n", e);
  }
  puts("done\n");
}
