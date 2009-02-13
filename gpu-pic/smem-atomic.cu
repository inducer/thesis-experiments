#include <stdio.h>

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

    oop = 0;
    #if 1
    if (n_pt >= bep)
      atomicAdd(&oop, 1);
    #endif
    if (oop == THREADS_PER_BLOCK)
      break;
  }
}

int main()
{
  puts("V69\n");
  dim3 grid(4098,100);
  dim3 block(384,1);
  sift<<<grid,block>>>();
  cudaError e = cudaThreadSynchronize();
  printf("ret: %d\n", e);
}
