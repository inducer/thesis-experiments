#include <stdio.h>

#define SIZE 400
#define USE_SHARED

typedef int value_type;

__global__ void atomic_test_shared(value_type *dest)
{
  __shared__ int x;
  x = 0;
  __syncthreads();
  atomicAdd(&x, 1);
  __syncthreads();
  dest[threadIdx.x] = x;
}

__global__ void atomic_test_global(int *dest)
{
  *dest = 0;
  __syncthreads();
  atomicAdd(dest, 1);
  __syncthreads();
  dest[threadIdx.x] = dest[0];
}

int main()
{
  value_type *data;
  cudaMalloc( (void**) &data, SIZE*sizeof(value_type));
  
#ifdef USE_SHARED
  puts("using shared 2\n");
  atomic_test_shared<<<1,SIZE>>>(data);
#else
  puts("using global\n");
  atomic_test_global<<<1,SIZE>>>(data);
#endif

  value_type h_data[SIZE];
  cudaThreadSynchronize();
  cudaMemcpy(h_data, data, SIZE*sizeof(value_type), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  for (int i = 0; i < SIZE; ++i)
    printf("%d ", h_data[i]);
  puts("\n");
  return 0;
}
