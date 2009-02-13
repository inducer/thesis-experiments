from __future__ import division
import numpy
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
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

    bbox_min -= 1e-10
    bbox_max += 1e-10
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
                BoxFloat(*el.bounding_box(discr.mesh.points)).enlarged(1e-10)):
            grid_els.setdefault(brick.index(el_box_idx), []).append(el.id)

    # find nodes in each block
    def get_brick_idx(point):
        try:
            return brick.index(brick.which_cell(point))
        except RuntimeError:
            return None

    from pytools import flatten
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

    for grid_idx, node_indices in grid_idx_to_nodes.iteritems():
        split_idx = brick.split_index(grid_idx)
        assert brick.index(split_idx) == grid_idx
        node_start = 0

        grid_bbox = brick.cell(split_idx)
        assert brick.point(split_idx) in grid_bbox

        for node in discr.nodes[node_indices]:
            assert node in grid_bbox

        while node_start < len(node_indices):
            yield BlockInfo(
                    bbox_min=grid_bbox.lower,
                    bbox_max=grid_bbox.upper,
                    node_indices=node_indices[node_start:node_start+nodes_per_block]
                    )
            node_start += nodes_per_block




def make_bboxes(block_info_records, dimensions, dtype):
    bir = block_info_records

    bboxes = numpy.empty((len(bir), dimensions, 2), dtype=dtype)

    for block_nr, block_info in enumerate(bir):
        bboxes[block_nr,:,0] = block_info.bbox_min
        bboxes[block_nr,:,1] = block_info.bbox_max

    return bboxes




def make_block_data(block_info_records, discr, dtype, threads_per_block):
    bir = block_info_records
    block_starts = numpy.empty((len(bir)+1,), dtype=numpy.uint32)
    block_nodes = []
    block_permut_src_indices = numpy.zeros((len(discr.nodes),), dtype=numpy.uint32)

    node_num = 0
    for block_nr, block_info in enumerate(bir):
        l = len(block_info.node_indices)
        block_starts[block_nr] = node_num
        block_nodes.extend(discr.nodes[block_info.node_indices])
        block_permut_src_indices[block_info.node_indices] = \
                threads_per_block*block_nr + numpy.arange(l, dtype=numpy.uint32)
        node_num += l

    block_starts[len(bir)] = node_num
    assert node_num == len(discr.nodes)

    return (block_starts, 
            numpy.array(block_nodes, dtype=dtype), 
            block_permut_src_indices)




def time_sift(pib):
    from hedge.backends.cuda.tools import int_ceiling

    threads_per_block = 384     
    sift_max_particles_per_block = 10000
    dtype = numpy.float32
    node_count = len(pib.discr.nodes)
    particle_count = len(pib.x_particle)
    split_count = int_ceiling(particle_count/sift_max_particles_per_block)
    print split_count, "splits"

    # partitioning ------------------------------------------------------------
    el_group, = pib.discr.element_groups
    ldis = el_group.local_discretization

    max_nodes_per_block = threads_per_block

    # data generation ---------------------------------------------------------
    block_info_records = list(make_blocks(pib.discr,  max_nodes_per_block))
    sift_block_count = len(block_info_records)

    bboxes = make_bboxes(block_info_records, pib.discr.dimensions, dtype)
    block_starts, block_nodes, block_permut_src_indices = \
        make_block_data(block_info_records, pib.discr, dtype, threads_per_block)

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
    block_permut_src_indices_gpu = gpuarray.to_gpu(block_permut_src_indices)

    # GPU code ----------------------------------------------------------------
    ctx = {
            "split_count": split_count,
            "sift_threads_per_block": threads_per_block,
            "sift_block_count": sift_block_count,
            "sift_max_particles_per_block": sift_max_particles_per_block,
            "xdim_channels": xdim_channels,
            "vdim_channels": vdim_channels,
            "axes": ["x", "y", "z", "w"],
            }
    ctx.update(pib.__dict__)

    from jinja2 import Template
    sift_tpl = Template("""
    // defines ----------------------------------------------------------------

    #define PARTICLE_LIST_SIZE 1000
    #define THREADS_PER_BLOCK {{ sift_threads_per_block }}
    #define BLOCK_NUM (blockIdx.x)
    #define PARTICLE_CHUNK_NUM (blockIdx.y)
    #define PARTICLE_CHUNK_COUNT (gridDim.y)
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

    // main kernel ------------------------------------------------------------

    __shared__ volatile unsigned intersecting_particle_count;
    __shared__ unsigned intersecting_particles[PARTICLE_LIST_SIZE];
    __shared__ float bbox_min[XDIM];
    __shared__ float bbox_max[XDIM];
    __shared__ unsigned out_of_particles_thread_count;
    __shared__ unsigned block_end_particle;

    extern "C" __global__ void sift(
      float *debugbuf,
      float *j, 
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

        intersecting_particle_count = 0;

        block_end_particle = min(
          (PARTICLE_CHUNK_NUM+1)*{{ sift_max_particles_per_block }},
          particle_count);

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
              unsigned idx_in_list = intersecting_particle_count++;
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
            break;

          if (intersecting_particle_count >= PARTICLE_LIST_SIZE)
            break;
        }

        out_of_particles_thread_count = 0;
        intersecting_particle_count = 0;
        #if 1
        if (n_particle >= block_end_particle)
          atomicAdd(&out_of_particles_thread_count, 1);
        #endif
        if (out_of_particles_thread_count == THREADS_PER_BLOCK)
          break;
      }
    }
    \n""", line_statement_prefix="##")

    smod_sift = cuda.SourceModule(
            sift_tpl.render(**ctx), 
            no_extern_c=True, keep=True)

    # sift preparation --------------------------------------------------------

    tex_x_particle = smod_sift.get_texref("tex_x_particle")
    tex_x_particle.set_format(cuda.array_format.FLOAT, xdim_channels)
    x_particle_gpu.bind_to_texref(tex_x_particle)

    tex_v_particle = smod_sift.get_texref("tex_v_particle")
    tex_v_particle.set_format(cuda.array_format.FLOAT, vdim_channels)
    v_particle_gpu.bind_to_texref(tex_v_particle)

    tex_bboxes = smod_sift.get_texref("tex_bboxes")
    bboxes_gpu.bind_to_texref(tex_bboxes)

    tex_block_starts = smod_sift.get_texref("tex_block_starts")
    block_starts_gpu.bind_to_texref_ext(tex_block_starts)

    tex_block_nodes = smod_sift.get_texref("tex_block_nodes")
    block_nodes_gpu.bind_to_texref_ext(tex_block_nodes, xdim_channels)

    debugbuf = gpuarray.zeros((
      min(20000, sift_block_count*5),), dtype=numpy.float32)

    sift = smod_sift.get_function("sift").prepare(
            "PPfffffI",
            block=(threads_per_block,1,1),
            texrefs=[tex_x_particle, tex_v_particle, tex_bboxes, 
              tex_block_starts, tex_block_nodes])
    print "stats: smem=%d regs=%d lmem=%d" % (
        sift.smem, sift.registers, sift.lmem)

    j_grid_gpu = gpuarray.zeros(
            (pib.vdim, split_count, threads_per_block*sift_block_count, ), 
            dtype=dtype)


    # launch ------------------------------------------------------------------

    from pyrticle.tools import PolynomialShapeFunction
    sf = PolynomialShapeFunction(pib.radius, pib.xdim)

    print "SIFT"
    sift.prepared_call(
            (sift_block_count, split_count),
            debugbuf.gpudata,
            j_grid_gpu.gpudata,
            pib.charge, pib.radius, pib.radius**2, sf.normalizer, 
            pib.radius,
            particle_count)
    cuda.Context.synchronize()

    if False:
        numpy.set_printoptions(linewidth=150, threshold=2**15, suppress=True)
        debugbuf = debugbuf.get()
        #print debugbuf[:5*sift_block_count].reshape((sift_block_count,5))
        print debugbuf[:3*sift_block_count]




class ParticleInfoBlock(Record):
    pass

def make_pib(particle_count):
    MM = 1e-3
    if True:
        from pyrticle.geometry import make_cylinder_with_fine_core
        mesh = make_cylinder_with_fine_core(
            r=25*MM, inner_r=2.5*MM,
            min_z=-50*MM, max_z=50*MM,
            max_volume_inner=10*MM**3,
            max_volume_outer=100*MM**3,
            radial_subdiv=10)
    else:
        from hedge.mesh import make_cylinder_mesh
        mesh = make_cylinder_mesh(radius=25*MM, height=50*MM,
                max_volume=300*MM**3)

    xdim = 3
    vdim = 3
    xdim_align = 4 
    vdim_align = 4 

    mesh_min, mesh_max = mesh.bounding_box()
    center = (mesh_min+mesh_max)*0.5

    x_particle = 0.03*numpy.random.randn(particle_count, xdim)
    x_particle[:,0] *= 0.1
    x_particle[:,1] *= 0.1

    for i in range(xdim):
        x_particle[:,i] += center[i]

    v_particle = numpy.zeros((particle_count, vdim))
    v_particle[:,0] = 1

    from hedge.backends.jit import Discretization
    discr = Discretization(mesh, order=4)

    return ParticleInfoBlock(
        mesh=mesh,
        discr=discr,
        xdim=xdim,
        vdim=vdim,
        x_particle=x_particle,
        v_particle=v_particle,
        charge=1,
        radius=0.000556605732511,
        #radius=50*MM,
        )

def main():
    print "V30"
    print "Making PIB"
    pib = make_pib(10**5)
    print "done"
    time_sift(pib)

if __name__ == "__main__":
    main()
