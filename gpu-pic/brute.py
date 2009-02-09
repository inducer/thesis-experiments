import numpy
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.tools




def time_brute(pib):
    devdata = pycuda.tools.DeviceData()

    xdim_channels = devdata.make_valid_tex_channel_count(pib.xdim)
    vdim_channels = devdata.make_valid_tex_channel_count(pib.vdim)

    node_count = len(pib.discr.nodes)
    particle_count = len(pib.x_particle)

    dtype = numpy.float32
    x_node = numpy.zeros((node_count, xdim_channels), dtype=dtype)
    x_node[:,:pib.xdim] = pib.discr.nodes
    x_node[:,pib.xdim:] = 0

    x_particle = numpy.zeros((particle_count, xdim_channels), dtype=dtype)
    x_particle[:particle_count, :pib.xdim] = pib.x_particle

    v_particle = numpy.zeros((particle_count, vdim_channels), dtype=dtype)
    v_particle[:particle_count, :pib.vdim] = pib.v_particle

    x_node_gpu = gpuarray.to_gpu(x_node)
    x_particle_gpu = gpuarray.to_gpu(x_particle)
    v_particle_gpu = gpuarray.to_gpu(v_particle)

    threads_per_block = 384     

    ctx = {}
    ctx.update(locals())
    ctx.update(pib.__dict__)

    smod = cuda.SourceModule("""
    #define THREADS_PER_BLOCK %(threads_per_block)d

    typedef float%(xdim_channels)d pos_align_vec;
    typedef float%(vdim_channels)d velocity_align_vec;
    typedef float%(xdim)d pos_vec;
    typedef float%(vdim)d velocity_vec;

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;
    texture<velocity_align_vec, 1, cudaReadModeElementType> tex_v_particle;
    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_node;

    // shortening and lengthening ---------------------------------------------
    template <class T>
    class shorten { };

    template <>
    struct shorten<float3> 
    { 
      typedef float3 target_type;
      __device__ static target_type call(float4 x)
      { return make_float3(x.x, x.y, x.z); }
    };

    template <class T>
    class lengthen { };

    template <>
    struct lengthen<float4> 
    { 
      typedef float4 target_type;
      __device__ static target_type call(float3 x)
      { return make_float4(x.x, x.y, x.z, 0); }
    };

    // main kernel ------------------------------------------------------------

    __shared__ pos_vec smem_x_particle[THREADS_PER_BLOCK];

    extern "C" __global__ void brute(
      float *debugbuf,
      velocity_align_vec *j, 
      float particle, float radius, float radius_squared, float normalizer, 
      unsigned node_count, unsigned particle_count)
    {
      unsigned node_id = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

      pos_vec x_node = shorten<pos_vec>::call(tex1Dfetch(tex_x_node, node_id));

      velocity_vec my_j;

      #pragma unroll 5
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
        j[node_id] = lengthen<velocity_align_vec>::call(my_j);
    }
    \n""" % ctx, no_extern_c=True, keep=True)

    tex_x_node = smod.get_texref("tex_x_node")
    tex_x_node.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_node_gpu.bind_to_texref(tex_x_node)

    tex_x_particle = smod.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_particle_gpu.bind_to_texref(tex_x_particle)

    tex_v_particle = smod.get_texref("tex_v_particle")
    tex_v_particle.set_format(cuda.array_format.FLOAT, vdim_channels)
    v_particle_gpu.bind_to_texref(tex_v_particle)

    debugbuf = gpuarray.zeros((threads_per_block,), dtype=numpy.float32)

    func = smod.get_function("brute").prepare(
            "PPffffII",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_node, tex_x_particle, tex_v_particle])
    print "stats: smem=%d regs=%d lmem=%d" % (
        func.smem, func.registers, func.lmem)

    start = cuda.Event()
    stop = cuda.Event()

    j_gpu = gpuarray.zeros((node_count, vdim_channels), dtype=dtype)

    from hedge.backends.cuda.tools import int_ceiling

    charge = 1
    radius = 0.000556605732511
    #radius = 50*MM

    from pyrticle.tools import PolynomialShapeFunction
    sf = PolynomialShapeFunction(radius, pib.xdim)

    start.record()
    func.prepared_call(
            (int_ceiling(node_count/threads_per_block), 1),
            debugbuf.gpudata,
            j_gpu.gpudata,
            charge, radius, radius**2, sf.normalizer,
            node_count, particle_count)
    stop.record()
    stop.synchronize()

    #print debugbuf

    from hedge.tools import to_obj_array
    j = to_obj_array(j_gpu.get().T)[:3]

    t = stop.time_since(start)*1e-3
    return particle_count/t, j
