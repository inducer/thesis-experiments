from __future__ import division
import numpy
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.tools




def time_sift():
    from hedge.backends.cuda.tools import int_ceiling

    particle_count = 10**5
    xdim = 3
    threads_per_block = 384     
    sift_block_count = 4098

    devdata = pycuda.tools.DeviceData()

    xdim_channels = devdata.make_valid_tex_channel_count(xdim)
    x_particle = numpy.zeros((particle_count, xdim_channels), dtype=numpy.float32)
    x_particle[:particle_count, :xdim] = numpy.random.randn(particle_count, xdim)

    x_particle_gpu = gpuarray.to_gpu(x_particle)

    # GPU code ----------------------------------------------------------------
    sift_code = """
    // defines ----------------------------------------------------------------

    #define THREADS_PER_BLOCK 384

    typedef float4 pos_align_vec;
    typedef float3 pos_vec;

    // textures ---------------------------------------------------------------

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;

    // main kernel ------------------------------------------------------------

    __shared__ unsigned out_of_particles_thread_count;
    __shared__ unsigned block_end_particle;

    extern "C" __global__ void sift(float *debugbuf)
    {
      if (threadIdx.x == 0)
      {
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
    x_particle_gpu.bind_to_texref_ext(tex_x_particle)

    debugbuf = gpuarray.zeros((
      min(20000, sift_block_count*5),), dtype=numpy.float32)

    sift = smod_sift.get_function("sift").prepare("P", 
        block=(threads_per_block,1,1),
        texrefs=[tex_x_particle])

    # launch ------------------------------------------------------------------

    print "LAUNCH"
    sift.prepared_call((sift_block_count, 100), debugbuf.gpudata)

    if False:
        numpy.set_printoptions(linewidth=150, threshold=2**15, suppress=True)
        debugbuf = debugbuf.get()
        #print debugbuf[:5*sift_block_count].reshape((sift_block_count,5))
        print debugbuf[:3*sift_block_count]




if __name__ == "__main__":
    print "V61"
    time_sift()
