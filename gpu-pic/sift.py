import numpy
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.tools




def make_bboxes(dimensions, dtype, mesh, blocks):
    bboxes = numpy.empty((len(blocks), dimensions, 2), dtype=dtype)

    from pytools import flatten
    for block_nr, block in enumerate(blocks):
        vertices = numpy.array([
            mesh.points[vi] 
            for el_nr in block
            for vi in mesh.elements[el_nr].vertex_indices])

        bboxes[block_nr,:,0] = numpy.minimum.reduce(vertices)
        bboxes[block_nr,:,1] = numpy.maximum.reduce(vertices)

        return bboxes




def make_gpu_blocks(blocks, discr):
    block_starts = numpy.empty((len(blocks)+1,), dtype=numpy.uint32)
    block_nodes = numpy.empty((len(discr.nodes),), dtype=numpy.uint32)

    node_num = 0
    for block_nr, block in enumerate(blocks):
        block_starts[block_nr] = node_num
        for el_nr in block:
            rng = discr.find_el_range(el_nr)
            l = rng.stop-rng.start
            block_nodes[node_num:node_num+l] = numpy.arange(rng.start, rng.stop)
            node_num += l

    block_starts[len(blocks)] = node_num

    return block_starts, block_nodes




def time_sift(pib):
    threads_per_block = 384     
    dtype = numpy.float32
    node_count = len(pib.discr.nodes)
    particle_count = len(pib.x_particle)

    # partitioning ------------------------------------------------------------
    el_group, = pib.discr.element_groups
    ldis = el_group.local_discretization

    elements_per_block = threads_per_block // ldis.node_count()

    from hedge.cuda.discretization import make_gpu_partition
    partition, blocks = make_gpu_partition(
            pib.mesh.element_adjacency_graph(), elements_per_block)

    # data generation ---------------------------------------------------------
    bboxes = make_bboxes(pib.xdim, dtype, pib.discr.mesh, blocks)
    block_starts, block_nodes = make_gpu_blocks(blocks, pib.discr)

    # To-GPU copy -------------------------------------------------------------
    devdata = pycuda.tools.DeviceData()

    xdim_channels = devdata.make_valid_tex_channel_count(pib.xdim)
    vdim_channels = devdata.make_valid_tex_channel_count(pib.vdim)

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

    bboxes_gpu = gpuarray.to_gpu(bboxes)
    block_starts_gpu = gpuarray.to_gpu(block_starts)
    block_nodes_gpu = gpuarray.to_gpu(block_nodes)

    # GPU code ----------------------------------------------------------------
    ctx = {"valid_threads": ldis.node_count()*elements_per_block}
    ctx.update(locals())
    ctx.update(pib.__dict__)

    smod = cuda.SourceModule("""
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

    // defines ----------------------------------------------------------------

    #define PARTICLE_LIST_SIZE 1000
    #define THREADS_PER_BLOCK %(threads_per_block)d
    #define BLOCK_NUM (blockIdx.x)
    #define XDIM %(xdim)d
    #define VALID_THREADS %(valid_threads)d
    #define BLOCK_START tex1Dfetch(tex_block_starts, BLOCK_NUM)
    #define NODE_ID tex1Dfetch(tex_block_nodes, BLOCK_START+threadIdx.x)

    typedef float%(xdim_channels)d pos_align_vec;
    typedef float%(vdim_channels)d velocity_align_vec;
    typedef float%(xdim)d pos_vec;
    typedef float%(vdim)d velocity_vec;

    // textures ---------------------------------------------------------------

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;
    texture<velocity_align_vec, 1, cudaReadModeElementType> tex_v_particle;
    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_node;
    texture<float, 1, cudaReadModeElementType> tex_bboxes;
    texture<unsigned, 1, cudaReadModeElementType> tex_block_starts;
    texture<unsigned, 1, cudaReadModeElementType> tex_block_nodes;

    // main kernel ------------------------------------------------------------

    __shared__ unsigned todo_particle_count;
    __shared__ unsigned todo_particles[PARTICLE_LIST_SIZE];
    __shared__ float bbox_min[XDIM];
    __shared__ float bbox_max[XDIM];
    __shared__ unsigned out_of_particles_thread_count;

    extern "C" __global__ void sift(
      float *debugbuf,
      velocity_align_vec *j, 
      float particle, float radius, float radius_squared, float normalizer, 
      unsigned node_count, unsigned particle_count)
    {
      if (threadIdx.x == 0)
      {
        // setup --------------------------------------------------------------
        out_of_particles_thread_count = 0;

        for (unsigned i = 0; i < XDIM; ++i)
        {
          bbox_min[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2);
          bbox_max[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2 + 1);
        }
      }
      __syncthreads();
      
      velocity_vec my_j;

      unsigned n_particle = threadIdx.x;
      while (n_particle < particle_count)
      {
        // sift ---------------------------------------------------------------
        todo_particle_count = 0;
        
        while (true)
        {
          if (n_particle < particle_count)
          {
            //#pragma unroll
            //for (unsigned repeats = 0; repeats < 4; ++repeats)
            {
              float3 x_particle = shorten<pos_vec>::call(
                tex1Dfetch(tex_x_particle, n_particle)); 

              int out_of_bbox_indicator =
                  signbit(x_particle.x - bbox_min[0])
                + signbit(bbox_max[0] - x_particle.x)
                + signbit(x_particle.y - bbox_min[1])
                + signbit(bbox_max[1] - x_particle.y)
                + signbit(x_particle.z - bbox_min[2])
                + signbit(bbox_max[2] - x_particle.z)
                ;
              if (out_of_bbox_indicator)
              {
                unsigned idx_in_list = atomicAdd(&todo_particle_count, 1)-1;
                if (idx_in_list < PARTICLE_LIST_SIZE)
                {
                  todo_particles[idx_in_list] = n_particle;
                  n_particle += THREADS_PER_BLOCK;
                }
              }
              else
                n_particle += THREADS_PER_BLOCK;
            }
          }
          else
          {
            // every thread should only contribute to out_of_particles_thread_count once.

            if (n_particle != unsigned(-1))
              atomicAdd(&out_of_particles_thread_count, 1);
            n_particle = unsigned(-1);
          }

          // loop end conditions

          __syncthreads();
          if (out_of_particles_thread_count == THREADS_PER_BLOCK)
            break;

          if (todo_particle_count >= PARTICLE_LIST_SIZE)
            break;

          __syncthreads();
        }

        // add up todo_particles ----------------------------------------------
        pos_vec x_node = shorten<pos_vec>::call(
          tex1Dfetch(tex_x_node, NODE_ID));

        if (threadIdx.x < VALID_THREADS)
        {
          for (unsigned block_particle_nr = 0;
               block_particle_nr < todo_particle_count;
               ++block_particle_nr)
          {
            unsigned particle_nr = todo_particles[block_particle_nr];
            pos_vec x_particle = shorten<pos_vec>::call(
              tex1Dfetch(tex_x_particle, particle_nr));
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

              velocity_vec v_particle = shorten<pos_vec>::call(
                tex1Dfetch(tex_v_particle, particle_nr));

              my_j.x += blob*v_particle.x;
              my_j.y += blob*v_particle.y;
              my_j.z += blob*v_particle.z;
            }
          }
        }
      }

      unsigned node_id = NODE_ID;
      if (node_id < node_count)
        j[node_id] = lengthen<velocity_align_vec>::call(my_j);
    }
    \n""" % ctx, no_extern_c=True, keep=True)

    # GPU invocation ----------------------------------------------------------

    tex_x_node = smod.get_texref("tex_x_node")
    tex_x_node.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_node_gpu.bind_to_texref(tex_x_node)

    tex_x_particle = smod.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_particle_gpu.bind_to_texref(tex_x_particle)

    tex_v_particle = smod.get_texref("tex_v_particle")
    tex_v_particle.set_format(cuda.array_format.FLOAT, vdim_channels)
    v_particle_gpu.bind_to_texref(tex_v_particle)

    tex_bboxes = smod.get_texref("tex_bboxes")
    tex_bboxes.set_flags(cuda.TRSF_READ_AS_INTEGER)
    bboxes_gpu.bind_to_texref(tex_bboxes)

    tex_block_starts = smod.get_texref("tex_block_starts")
    tex_block_starts.set_flags(cuda.TRSF_READ_AS_INTEGER)
    block_starts_gpu.bind_to_texref(tex_block_starts)

    tex_block_nodes = smod.get_texref("tex_block_nodes")
    tex_block_nodes.set_flags(cuda.TRSF_READ_AS_INTEGER)
    block_nodes_gpu.bind_to_texref(tex_block_nodes)

    debugbuf = gpuarray.zeros((threads_per_block,), dtype=numpy.float32)

    func = smod.get_function("sift").prepare(
            "PPffffII",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_node, tex_x_particle, tex_v_particle, tex_bboxes, tex_block_starts, tex_block_nodes])
    print "stats: smem=%d regs=%d lmem=%d" % (
        func.smem, func.registers, func.lmem)

    start = cuda.Event()
    stop = cuda.Event()

    j_gpu = gpuarray.zeros((node_count, vdim_channels), dtype=dtype)

    from hedge.cuda.tools import int_ceiling

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
