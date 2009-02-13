from __future__ import division
import numpy
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.tools




def make_bboxes(block_count, dimensions, dtype):
    block_count = 4098
    n = block_count**(1/3)
    dx = 2/n
    
    bboxes = numpy.empty((block_count, dimensions, 2), dtype=dtype)

    block_nr = 0
    for xi in numpy.arange(-1, 1, dx):
        for yi in numpy.arange(-1, 1, dx):
            for zi in numpy.arange(-1, 1, dx):
                pt = numpy.array([xi,yi,zi])
                bboxes[block_nr,:,0] = pt-dx/2
                bboxes[block_nr,:,1] = pt+dx/2
                if block_nr >= block_count:
                    break

    return bboxes




def time_sift():
    from hedge.backends.cuda.tools import int_ceiling

    particle_count = 10**5
    xdim = 3
    pib_x_particle = numpy.random.randn(particle_count, xdim)

    threads_per_block = 384     
    dtype = numpy.float32

    # data generation ---------------------------------------------------------
    sift_block_count = 4098

    bboxes = make_bboxes(sift_block_count, xdim, dtype)

    # To-GPU copy -------------------------------------------------------------
    devdata = pycuda.tools.DeviceData()

    xdim_channels = devdata.make_valid_tex_channel_count(xdim)
    x_particle = numpy.zeros((particle_count, xdim_channels), dtype=dtype)
    x_particle[:particle_count, :xdim] = pib_x_particle

    x_particle_gpu = gpuarray.to_gpu(x_particle)
    bboxes_gpu = gpuarray.to_gpu(bboxes)

    # GPU code ----------------------------------------------------------------
    sift_code = """
    // defines ----------------------------------------------------------------

    #define PARTICLE_LIST_SIZE 1000
    #define THREADS_PER_BLOCK 384
    #define BLOCK_NUM (blockIdx.x)
    #define XDIM 3

    typedef float4 pos_align_vec;
    typedef float3 pos_vec;

    // textures ---------------------------------------------------------------

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;
    texture<float, 1, cudaReadModeElementType> tex_bboxes;

    // main kernel ------------------------------------------------------------

    __shared__ float bbox_min[XDIM];
    __shared__ float bbox_max[XDIM];
    __shared__ unsigned out_of_particles_thread_count;
    __shared__ unsigned block_end_particle;

    extern "C" __global__ void sift(
      float *debugbuf,
      float particle, float radius, float radius_squared, float normalizer, 
      float box_pad,
      unsigned particle_count
      )
    {
      if (threadIdx.x == 0)
      {
        // setup --------------------------------------------------------------
        for (unsigned i = 0; i < XDIM; ++i)
        {
          bbox_min[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2) - box_pad;
          bbox_max[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2 + 1) + box_pad;
        }

        block_end_particle = 10000;
      }
      __syncthreads();
      
      unsigned n_particle = threadIdx.x;

      unsigned blah, blubb;
      for (blah = 0; blah < 20; ++blah)
      {
        // sift ---------------------------------------------------------------
        
        for (blubb = 0; blubb < 20; ++blubb)
        {
          if (n_particle < block_end_particle)
          {
            float4 x_particle = tex1Dfetch(tex_x_particle, n_particle); 

            int out_of_bbox_indicator = sin(x_particle.x*500) > 0.1;
            if (!out_of_bbox_indicator)
            {
              n_particle += THREADS_PER_BLOCK;
            }
            else
              n_particle += THREADS_PER_BLOCK;
          }
          else
            break;

        }

        out_of_particles_thread_count = 0;
        #if 1
        if (n_particle >= block_end_particle)
          atomicAdd(&out_of_particles_thread_count, 1);
        #endif
        if (out_of_particles_thread_count == THREADS_PER_BLOCK)
          break;
      }
    }
    \n"""

    smod_sift = cuda.SourceModule(sift_code, no_extern_c=True, keep=True)

    # sift preparation --------------------------------------------------------

    tex_x_particle = smod_sift.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_particle_gpu.bind_to_texref(tex_x_particle)

    tex_bboxes = smod_sift.get_texref("tex_bboxes")
    bboxes_gpu.bind_to_texref(tex_bboxes)

    debugbuf = gpuarray.zeros((
      min(20000, sift_block_count*5),), dtype=numpy.float32)

    sift = smod_sift.get_function("sift").prepare(
            "PfffffI",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_particle, tex_bboxes])

    # launch ------------------------------------------------------------------

    print "LAUNCH"
    pib_radius = 0.3
    sift.prepared_call(
            (sift_block_count, 100),
            debugbuf.gpudata,
            1, pib_radius, pib_radius**2, 1, 
            pib_radius,
            particle_count)
    cuda.Context.synchronize()

    if False:
        numpy.set_printoptions(linewidth=150, threshold=2**15, suppress=True)
        debugbuf = debugbuf.get()
        #print debugbuf[:5*sift_block_count].reshape((sift_block_count,5))
        print debugbuf[:3*sift_block_count]




if __name__ == "__main__":
    print "V54"
    time_sift()
