from __future__ import division
import numpy
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.tools
from pytools import Record




class BlockInfo(Record):
    pass



def make_blocks(discr, nodes_per_block):
    from hedge.discretization import ones_on_volume
    mesh_volume = discr.integral(ones_on_volume(discr))
    dx =  (mesh_volume / len(discr))**(1/discr.dimensions) * 4

    mesh = discr.mesh
    bbox_min, bbox_max = mesh.bounding_box()

    bbox_size = bbox_max-bbox_min
    dims = numpy.asarray(bbox_size/dx, dtype=numpy.int32)

    from pyrticle._internal import Brick, BoxFloat
    brick = Brick(number=0,
            start_index=0,
            stepwidths=bbox_size/dims,
            origin=bbox_min,
            dimensions=dims)

    # find elements that overlap each grid index
    grid_els = {}
    for el in discr.mesh.elements:
        for el_box_idx in brick.get_iterator(
                BoxFloat(*el.bounding_box(discr.mesh.points))):
            grid_els.setdefault(brick.index(el_box_idx), []).append(el.id)

    # find nodes in each block
    from pytools import flatten
    def get_brick_idx(point):
        try:
            return brick.index(brick.which_cell(point))
        except RuntimeError:
            return None

    grid_idx_to_nodes = {}
    for grid_idx, element_ids in grid_els.iteritems():
        nodes = list(flatten([
            [ni for ni in range(
                discr.find_el_range(el_id).start,
                discr.find_el_range(el_id).stop)
                if get_brick_idx(discr.nodes[ni]) == grid_idx]
            for el_id in element_ids]))
        if nodes:
            grid_idx_to_nodes[grid_idx] = nodes

    for grid_idx, nodes in grid_idx_to_nodes.iteritems():
        node_start = 0
        while node_start < len(nodes):
            some_node = nodes[node_start]
            multi_idx = brick.which_cell(discr.nodes[some_node])

            yield BlockInfo(
                    bbox_min=brick.origin + multi_idx*brick.stepwidths,
                    bbox_max=brick.origin + (multi_idx+1)*brick.stepwidths,
                    nodes=nodes[node_start:node_start+nodes_per_block],
                    )
            node_start += nodes_per_block




def make_bboxes(block_info_records, dimensions, pad, dtype):
    bir = block_info_records

    bboxes = numpy.empty((len(bir), dimensions, 2), dtype=dtype)

    for block_nr, block_info in enumerate(bir):
        bboxes[block_nr,:,0] = block_info.bbox_min - pad
        bboxes[block_nr,:,1] = block_info.bbox_max + pad

    return bboxes




def make_block_data(block_info_records, discr, dtype):
    bir = block_info_records
    block_starts = numpy.empty((len(bir)+1,), dtype=numpy.uint32)
    block_nodes = numpy.empty((len(discr.nodes), discr.dimensions), dtype=dtype)
    block_node_indices = numpy.empty((len(discr.nodes), ), dtype=numpy.uint32)

    node_num = 0
    for block_nr, block_info in enumerate(bir):
        block_starts[block_nr] = node_num
        l = len(block_info.nodes)
        block_nodes[node_num:node_num+l] = discr.nodes[block_info.nodes]
        block_node_indices[node_num:node_num+l] = block_info.nodes
        node_num += l
        assert node_num < len(discr.nodes)

    block_starts[len(bir)] = node_num

    return block_starts, block_nodes, block_node_indices




def time_sift(pib):
    threads_per_block = 384     
    dtype = numpy.float32
    node_count = len(pib.discr.nodes)
    particle_count = len(pib.x_particle)

    charge = 1
    radius = 0.000556605732511
    #radius = 50*MM

    # partitioning ------------------------------------------------------------
    el_group, = pib.discr.element_groups
    ldis = el_group.local_discretization

    max_nodes_per_block = threads_per_block

    # data generation ---------------------------------------------------------
    block_info_records = list(make_blocks(pib.discr,  max_nodes_per_block))
    bboxes = make_bboxes(block_info_records, pib.discr.dimensions, radius, dtype)
    block_starts, block_nodes, block_node_indices = \
        make_block_data(block_info_records, pib.discr, dtype)

    # To-GPU copy -------------------------------------------------------------
    devdata = pycuda.tools.DeviceData()

    xdim_channels = devdata.make_valid_tex_channel_count(pib.xdim)
    vdim_channels = devdata.make_valid_tex_channel_count(pib.vdim)

    block_nodes_tdata = numpy.zeros(
            (block_nodes.shape[0], xdim_channels), dtype=dtype)
    block_nodes_tdata[:,:pib.xdim] = block_nodes

    x_particle = numpy.zeros((particle_count, xdim_channels), dtype=dtype)
    x_particle[:particle_count, :pib.xdim] = pib.x_particle

    v_particle = numpy.zeros((particle_count, vdim_channels), dtype=dtype)
    v_particle[:particle_count, :pib.vdim] = pib.v_particle

    x_particle_gpu = gpuarray.to_gpu(x_particle)
    v_particle_gpu = gpuarray.to_gpu(v_particle)

    bboxes_gpu = gpuarray.to_gpu(bboxes)
    block_starts_gpu = gpuarray.to_gpu(block_starts)
    block_nodes_gpu = gpuarray.to_gpu(block_nodes_tdata)
    block_node_indices_gpu = gpuarray.to_gpu(block_node_indices)

    # GPU code ----------------------------------------------------------------
    ctx = {}
    ctx.update(locals())
    ctx.update(pib.__dict__)

    from jinja2 import Template
    source_tpl = Template("""
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
    #define THREADS_PER_BLOCK {{ threads_per_block }}
    #define BLOCK_NUM (blockIdx.x)
    #define XDIM {{ xdim }}
    #define BLOCK_START tex1Dfetch(tex_block_starts, BLOCK_NUM)
    #define BLOCK_END tex1Dfetch(tex_block_starts, BLOCK_NUM+1)

    typedef float{{ xdim_channels }} pos_align_vec;
    typedef float{{ vdim_channels }} velocity_align_vec;
    typedef float{{ xdim }} pos_vec;
    typedef float{{ vdim }} velocity_vec;

    // textures ---------------------------------------------------------------

    texture<pos_align_vec, 1, cudaReadModeElementType> tex_x_particle;
    texture<velocity_align_vec, 1, cudaReadModeElementType> tex_v_particle;
    texture<float, 1, cudaReadModeElementType> tex_bboxes;
    texture<unsigned, 1, cudaReadModeElementType> tex_block_starts;
    texture<pos_align_vec, 1, cudaReadModeElementType> tex_block_nodes;
    texture<unsigned, 1, cudaReadModeElementType> tex_block_node_indices;

    // main kernel ------------------------------------------------------------

    __shared__ unsigned intersecting_particle_count;
    __shared__ unsigned intersecting_particles[PARTICLE_LIST_SIZE];
    __shared__ float bbox_min[XDIM];
    __shared__ float bbox_max[XDIM];
    __shared__ unsigned out_of_particles_thread_count;

    extern "C" __global__ void sift(
      float *debugbuf,
      velocity_vec *j, 
      float particle, float radius, float radius_squared, float normalizer, 
      unsigned node_count, unsigned particle_count
      )
    {
      if (threadIdx.x == 0)
      {
        // setup --------------------------------------------------------------
        for (unsigned i = 0; i < XDIM; ++i)
        {
          bbox_min[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2);
          bbox_max[i] = tex1Dfetch(tex_bboxes, (BLOCK_NUM*XDIM + i) * 2 + 1);
        }

        out_of_particles_thread_count = 0;
        intersecting_particle_count = 0;
      }
      __syncthreads();
      
      velocity_vec my_j;
      my_j.x = 0;
      my_j.y = 0;
      my_j.z = 0;

      unsigned n_particle = threadIdx.x;
      while (true)
      {
        // sift ---------------------------------------------------------------
        
        while (true)
        {
          if (n_particle < particle_count)
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
            if (!out_of_bbox_indicator)
            {
              unsigned idx_in_list = atomicAdd(&intersecting_particle_count, 1);
              if (idx_in_list < PARTICLE_LIST_SIZE)
              {
                intersecting_particles[idx_in_list] = n_particle;
                n_particle += THREADS_PER_BLOCK;
              }
            }
            else
              n_particle += THREADS_PER_BLOCK;
          }
          else
          {
            // every thread should only contribute to out_of_particles_thread_count once.

            if (n_particle != UINT_MAX)
              atomicAdd(&out_of_particles_thread_count, 1);
            n_particle = UINT_MAX;
          }

          // loop end conditions

          __syncthreads();
          if (out_of_particles_thread_count == THREADS_PER_BLOCK)
            break;

          if (intersecting_particle_count >= PARTICLE_LIST_SIZE)
            break;
          __syncthreads();
        }

        // add up intersecting_particles --------------------------------------
        if (threadIdx.x < BLOCK_END-BLOCK_START)
        {
          pos_vec x_node = shorten<pos_vec>::call(
              tex1Dfetch(tex_block_nodes, BLOCK_START+threadIdx.x));

          unsigned block_particle_count = min(
            intersecting_particle_count,
            PARTICLE_LIST_SIZE);

          if (threadIdx.x == 0)
            debugbuf[blockIdx.x*5+1] += intersecting_particle_count;

          for (unsigned block_particle_nr = 0;
               block_particle_nr < block_particle_count;
               ++block_particle_nr)
          {
            unsigned particle_nr = intersecting_particles[block_particle_nr];
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

        if (out_of_particles_thread_count == THREADS_PER_BLOCK)
          break;

        __syncthreads();
        intersecting_particle_count = 0;
        __syncthreads();
      }

      if (threadIdx.x < BLOCK_END-BLOCK_START)
        j[tex1Dfetch(tex_block_node_indices, BLOCK_START+threadIdx.x)] = my_j;
    }
    \n""", line_statement_prefix="##")

    smod = cuda.SourceModule(
            source_tpl.render(**ctx), 
            no_extern_c=True, keep=True)

    # GPU invocation ----------------------------------------------------------

    tex_x_particle = smod.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_particle_gpu.bind_to_texref(tex_x_particle)

    tex_v_particle = smod.get_texref("tex_v_particle")
    tex_v_particle.set_format(cuda.array_format.FLOAT, vdim_channels)
    v_particle_gpu.bind_to_texref(tex_v_particle)

    tex_bboxes = smod.get_texref("tex_bboxes")
    bboxes_gpu.bind_to_texref(tex_bboxes)

    tex_block_starts = smod.get_texref("tex_block_starts")
    tex_block_starts.set_flags(cuda.TRSF_READ_AS_INTEGER)
    block_starts_gpu.bind_to_texref(tex_block_starts)

    tex_block_nodes = smod.get_texref("tex_block_nodes")
    tex_block_nodes.set_format(cuda.array_format.FLOAT, xdim_channels)
    block_nodes_gpu.bind_to_texref(tex_block_nodes)

    tex_block_nodes = smod.get_texref("tex_block_nodes")
    tex_block_nodes.set_format(cuda.array_format.FLOAT, xdim_channels)
    block_nodes_gpu.bind_to_texref(tex_block_nodes)

    tex_block_node_indices = smod.get_texref("tex_block_node_indices")
    tex_block_node_indices.set_flags(cuda.TRSF_READ_AS_INTEGER)
    block_starts_gpu.bind_to_texref(tex_block_node_indices)

    from hedge.backends.cuda.tools import int_ceiling
    block_count = len(block_info_records)

    debugbuf = gpuarray.zeros((block_count*5,), dtype=numpy.float32)

    func = smod.get_function("sift").prepare(
            "PPffffIIPP",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_particle, tex_v_particle, tex_bboxes, 
              tex_block_starts, tex_block_nodes, tex_block_node_indices])
    print "stats: smem=%d regs=%d lmem=%d" % (
        func.smem, func.registers, func.lmem)

    start = cuda.Event()
    stop = cuda.Event()

    j_gpu = gpuarray.zeros((node_count, pib.vdim), dtype=dtype)

    from pyrticle.tools import PolynomialShapeFunction
    sf = PolynomialShapeFunction(pib.radius, pib.xdim)

    print "LAUNCHING"
    start.record()
    func.prepared_call(
            (block_count, 1),
            debugbuf.gpudata,
            j_gpu.gpudata,
            pib.charge, pib.radius, pib.radius**2, sf.normalizer,
            node_count, particle_count,
            cuda.mem_alloc(block_count*4), # out_of_particles_thread_count
            cuda.mem_alloc(block_count*4), # intersecting_particle_count
            )
    stop.record()
    stop.synchronize()

    if False:
        numpy.set_printoptions(linewidth=150, threshold=2**15, suppress=True)
        debugbuf = debugbuf.get()
        debugbuf[debugbuf > particle_count * 2] = -1
        print debugbuf[:5*block_count].reshape((block_count,5))

    from hedge.tools import to_obj_array
    j = to_obj_array(j_gpu.get().T)[:3]

    t = stop.time_since(start)*1e-3
    return particle_count/t, j

