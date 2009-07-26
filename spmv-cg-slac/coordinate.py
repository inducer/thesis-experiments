COO_FLAT_KERNEL_TEMPLATE = """
template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE, bool UseCache>
__global__ void
spmv_coo_flat_kernel(const IndexType num_nonzeros,
                     const IndexType interval_size,
                     const IndexType * I, 
                     const IndexType * J, 
                     const ValueType * V, 
                     const ValueType * x, 
                           ValueType * y)
{
    __shared__ IndexType idx[BLOCK_SIZE];
    __shared__ ValueType val[BLOCK_SIZE];
    __shared__ IndexType carry_idx[BLOCK_SIZE / 32];
    __shared__ ValueType carry_val[BLOCK_SIZE / 32];

    const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;     // global thread index
    const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);               // thread index within the warp
    const IndexType warp_id     = thread_id   / WARP_SIZE;                   // global warp index
    const IndexType warp_lane   = threadIdx.x / WARP_SIZE;                   // warp index within the CTA

    const IndexType begin = warp_id * interval_size + thread_lane;           // thread's offset into I,J,V
    const IndexType end   = min(begin + interval_size, num_nonzeros);        // end of thread's work

    if(begin >= end) return;                                                 // warp has no work to do

    const IndexType first_idx = I[warp_id * interval_size];                  // first row of this warp's interval

    if (thread_lane == 0){
        carry_idx[warp_lane] = first_idx; 
        carry_val[warp_lane] = 0;
    }
    
    for(IndexType n = begin; n < end; n += WARP_SIZE){
        idx[threadIdx.x] = I[n];                                             // row index
        val[threadIdx.x] = V[n] * fetch_x<UseCache>(J[n], x);                // val = A[row,col] * x[col] 

        if (thread_lane == 0){
            if(idx[threadIdx.x] == carry_idx[warp_lane])
                val[threadIdx.x] += carry_val[warp_lane];                    // row continues into this warp's span
            else if(carry_idx[warp_lane] != first_idx)
                y[carry_idx[warp_lane]] += carry_val[warp_lane];             // row terminated, does not span boundary
            else
                atomicAdd(y + carry_idx[warp_lane], carry_val[warp_lane]);   // row terminated, but spans iter-warp boundary
        }

        // segmented reduction in shared memory
        if( thread_lane >=  1 && idx[threadIdx.x] == idx[threadIdx.x - 1] ) { val[threadIdx.x] += val[threadIdx.x -  1]; EMUSYNC; } 
        if( thread_lane >=  2 && idx[threadIdx.x] == idx[threadIdx.x - 2] ) { val[threadIdx.x] += val[threadIdx.x -  2]; EMUSYNC; }
        if( thread_lane >=  4 && idx[threadIdx.x] == idx[threadIdx.x - 4] ) { val[threadIdx.x] += val[threadIdx.x -  4]; EMUSYNC; }
        if( thread_lane >=  8 && idx[threadIdx.x] == idx[threadIdx.x - 8] ) { val[threadIdx.x] += val[threadIdx.x -  8]; EMUSYNC; }
        if( thread_lane >= 16 && idx[threadIdx.x] == idx[threadIdx.x -16] ) { val[threadIdx.x] += val[threadIdx.x - 16]; EMUSYNC; }

        if( thread_lane == 31 ) {
            carry_idx[warp_lane] = idx[threadIdx.x];                         // last thread in warp saves its results
            carry_val[warp_lane] = val[threadIdx.x];
        }
        else if ( idx[threadIdx.x] != idx[threadIdx.x+1] ) {                 // row terminates here
            if(idx[threadIdx.x] != first_idx)
                y[idx[threadIdx.x]] += val[threadIdx.x];                     // row terminated, does not span inter-warp boundary
            else
                atomicAdd(y + idx[threadIdx.x], val[threadIdx.x]);           // row terminated, but spans iter-warp boundary
        }
        
    }

    // final carry
    if(thread_lane == 31){
        atomicAdd(y + carry_idx[warp_lane], carry_val[warp_lane]); 
    }
}
"""



COO_SERIAL_




class CoordinateSpMV:
    def __init__(self, mat, dtype):
        self.dtype = numpy.dtype(dtype)

        from scipy.sparse import coo_matrix
        self.coo_mat = coo_matrix(mat, dtype=self.dtype)

        print dir(self.coo_mat)

    def get_kernel(self):
