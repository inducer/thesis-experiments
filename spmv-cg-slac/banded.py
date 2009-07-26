class BandedSpMV:
    def __init__(self, mat, dtype=numpy.float32):
        from scipy.sparse import csr_matrix
        csr_mat = csr_matrix(mat)

        h, w = self.shape = csr_mat.shape

        # pass 1: find bandwidth
        self.left_bw = 0
        self.right_bw = 0
        print "pass 1"
        for i in xrange(h):
            for idx in range(csr_mat.indptr[i], csr_mat.indptr[i+1]):
                j = csr_mat.indices[idx]
                if i > j:
                    left_bw = max(left_bw, i-j)
                else:
                    right_bw = max(right_bw, j-i)

        print "bandwidths:", self.left_bw, self.right_bw

        # allocate storage
        from pycuda.tools import DeviceData
        devdata = DeviceData()
        self.total_bw = devdata.align_dtype(
                self.left_bw+self.right_bw+1,
                self.dtype.itemsize)

        self.data = numpy.zeros(
                (h, self.total_bw),
                dtype=self.dtype)

        # pass 2: populate storage
        for i in xrange(h):
            for idx in range(csr_mat.indptr[i], csr_mat.indptr[i+1]):
                j = csr_mat.indices[idx]
                a_ij = csr_mat.data[idx]
                self.data[i,j-i-left_bw] = a_ij

        self.gpu_data = gpuarray.to_gpu(self.data)

        self.block_size = 128

    @memoize_method
    def get_kernel(self):
        from pycuda.tools import dtype_to_ctype
        mod = SourceModule("""
            typedef int32_t index_type;
            typedef %(value_type)s value_type;

            #define BLOCK_SIZE %(block_size)d
            #define FETCH_GROUP_SIZE %(fetch_group_size)d

            __global__ void
            spmv_banded(const index_type num_rows, 
                            const index_type num_cols, 
                            const index_type left_bw,
                            const index_type total_bw,
                            const value_type *banded_data,
                            const value_type * x, 
                                  value_type * y)
            {
                __shared__ int offsets[BLOCK_SIZE];

                int fetch_start = max(0, 

                if(threadIdx.x < num_diags)
                    offsets[threadIdx.x] = diag_offsets[threadIdx.x];

                __syncthreads();

                const int row = large_grid_thread_id();

                if(row >= num_rows){ return; }

                value_type sum = y[row];
                diag_data += row;

                for(index_type n = 0; n < num_diags; n++){
                    const int col = row + offsets[n];

                    if(col >= 0 && col < num_cols){
                        const value_type A_ij = *diag_data;
                        sum += A_ij * fetch_x<UseCache>(col, x);
                    }

                    diag_data += stride;
                }

                y[row] = sum;
            }""" % {
                "block_size": self.block_size,
                "value_type": dtype_to_ctype(self.gpu_data.dtype)
                })

    def __call__(self, x):
        pass





