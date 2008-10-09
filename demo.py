import numpy
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray




def time_brute(particle_count):
    node_count = 860060
    xdim = 3
    vdim = 3
    xdim_align = 4 
    vdim_align = 4 

    dtype = numpy.float32
    x_node = numpy.random.randn(node_count, xdim_align).astype(dtype)
    x_particle = numpy.random.randn(particle_count, xdim_align).astype(dtype)
    v_particle = numpy.random.randn(particle_count, xdim_align).astype(dtype)

    x_node_gpu = gpuarray.to_gpu(x_node)
    x_particle_gpu = gpuarray.to_gpu(x_particle)
    v_particle_gpu = gpuarray.to_gpu(v_particle)

    threads_per_block = 256

    smod = cuda.SourceModule("""
    #define THREADS_PER_BLOCK %(threads_per_block)d

    typedef float%(xdim_align)d pos_align_vec;
    typedef float%(vdim_align)d velocity_align_vec;
    typedef float%(xdim)d pos_vec;
    typedef float%(vdim)d velocity_vec;

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;
    texture<velocity_align_vec, 1, cudaReadModeElementType> tex_v_particle;
    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_node;

    // shortening -------------------------------------------------------------
    template <class T>
    class shorten { };

    template <>
    struct shorten<float3> 
    { 
      typedef float3 target_type;
      __device__ static target_type call(float4 x)
      { return make_float3(x.x, x.y, x.z); }
    };

    // main kernel ------------------------------------------------------------

    __shared__ pos_vec smem_x_particle[THREADS_PER_BLOCK];

    extern "C" __global__ void brute(
      float *debugbuf,
      velocity_vec *j, 
      float particle, float radius, float radius_squared, float normalizer,
      unsigned node_count, unsigned particle_count)
    {
      unsigned node_id = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

      pos_vec x_node = shorten<pos_vec>::call(tex1Dfetch(tex_x_node, node_id));

      velocity_vec my_j;

      for (unsigned n_particle_base = 0; n_particle_base < particle_count; n_particle_base += THREADS_PER_BLOCK)
      {
        __syncthreads();
        smem_x_particle[threadIdx.x] = shorten<pos_vec>::call(tex1Dfetch(tex_x_particle, n_particle_base+threadIdx.x));
        __syncthreads();

        unsigned block_particle_count = THREADS_PER_BLOCK;
        if (n_particle_base + block_particle_count >= particle_count)
          block_particle_count = particle_count - n_particle_base;

        for (unsigned n_particle_local = 0; 
            n_particle_local < block_particle_count; 
            ++n_particle_local)
        {
          pos_vec x_particle = smem_x_particle[n_particle_local];
          pos_vec x_rel = make_float3(
              x_particle.x-x_node.x, 
              x_particle.y-x_node.y, 
              x_particle.z-x_node.z);
          float r_squared = 
              x_rel.x*x_rel.x
            + x_rel.y*x_rel.y
            + x_rel.z*x_rel.z;

          if (r_squared < radius_squared)
          {
            float z = radius-r_squared/radius;
            float blob = particle * normalizer * z*z;

            velocity_vec v_particle = shorten<pos_vec>::call(tex1Dfetch(tex_v_particle, n_particle_base+n_particle_local));

            my_j.x += blob*v_particle.x;
            my_j.y += blob*v_particle.y;
            my_j.z += blob*v_particle.z;
          }
        }
      }

      if (node_id < node_count)
        j[node_id] = my_j;
    }
    \n""" % locals(), no_extern_c=True, keep=True)

    tex_x_node = smod.get_texref("tex_x_node")
    tex_x_node.set_format(cuda.array_format.FLOAT, xdim_align)
    tex_x_particle = smod.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_align)
    tex_v_particle = smod.get_texref("tex_v_particle")
    tex_v_particle.set_format(cuda.array_format.FLOAT, vdim_align)

    x_node_gpu.bind_to_texref(tex_x_node)
    x_particle_gpu.bind_to_texref(tex_x_particle)
    v_particle_gpu.bind_to_texref(tex_v_particle)

    debugbuf = gpuarray.zeros((threads_per_block,), dtype=numpy.float32)

    func = smod.get_function("brute").prepare(
            "PPffffII",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_particle, tex_v_particle])
    print "stats: smem=%d regs=%d lmem=%d" % (
        func.smem, func.registers, func.lmem)

    start = cuda.Event()
    stop = cuda.Event()

    j = gpuarray.zeros((node_count, vdim_align), dtype=dtype)

    from hedge.cuda.tools import int_ceiling

    charge = 1
    radius = 0.1

    from pyrticle.tools import PolynomialShapeFunction
    sf = PolynomialShapeFunction(radius, xdim)

    start.record()
    func.prepared_call(
            (int_ceiling(node_count/threads_per_block), 1),
            debugbuf.gpudata,
            j.gpudata,
            charge, radius, radius**2, sf.normalizer,
            node_count, particle_count)
    stop.record()
    stop.synchronize()

    #print debugbuf

    t = stop.time_since(start)*1e-3
    return particle_count/t

def main():
    #for particle_count in [100*1000, 1000*1000, 10*1000*1000]:
    for particle_count in [100*1000, 1000*1000]:
        rate = time_brute(particle_count)
        print particle_count, rate

if __name__ == "__main__":
    main()
