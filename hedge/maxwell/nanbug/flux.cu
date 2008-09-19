extern "C" {
struct flux_header
{
  short unsigned int same_facepairs_end;
  short unsigned int diff_facepairs_end;
  short unsigned int bdry_facepairs_end;
  short unsigned int zero_facepairs_count;
};
struct face_pair
{
  float h;
  float order;
  float face_jacobian;
  float normal[3];
  unsigned int a_base;
  unsigned int b_base;
  short unsigned int a_ilist_index;
  short unsigned int b_ilist_index;
  short unsigned int b_write_ilist_index;
  unsigned char boundary_bitmap;
  unsigned char pad;
  short unsigned int a_dest;
  short unsigned int b_dest;
};

#define DIMENSIONS 3
#define DOFS_PER_EL 35
#define DOFS_PER_FACE 15

#define CONCURRENT_FACES 15
#define BLOCK_MB_COUNT 2

#define FACEDOF_NR threadIdx.x
#define BLOCK_FACE threadIdx.y

#define THREAD_NUM (FACEDOF_NR + BLOCK_FACE*DOFS_PER_FACE)
#define THREAD_COUNT (DOFS_PER_FACE*CONCURRENT_FACES)
#define COALESCING_THREAD_COUNT (THREAD_COUNT & ~0xf)

#define DATA_BLOCK_SIZE 1344
#define ALIGNED_FACE_DOFS_PER_MB 240
#define ALIGNED_FACE_DOFS_PER_BLOCK (ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT)


#define INDEX_LISTS_LENGTH 360
typedef unsigned char index_list_entry_t;

struct flux_data
{
  flux_header header;
  face_pair facepairs[29];
  short unsigned int zero_face_indices[12];
};

__shared__ index_list_entry_t smem_index_lists[INDEX_LISTS_LENGTH];
__shared__ flux_data data;
__shared__ float smem_fluxes_on_faces[ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT];

__global__ void apply_flux(float *debugbuf, unsigned char *gmem_data)
{
  /* load face_pair data */
  {
    unsigned int const *load_base = (unsigned int *) (gmem_data + blockIdx.x*DATA_BLOCK_SIZE);
    for (unsigned word_nr = THREAD_NUM; word_nr*sizeof(unsigned int) < (sizeof(flux_data)); word_nr += COALESCING_THREAD_COUNT)
      ((unsigned int *) (&data))[word_nr] = load_base[word_nr];
  }
  
  __syncthreads();
  
  /* compute the fluxes */
  short unsigned int fpair_nr = BLOCK_FACE;
  /* fluxes for dual-sided (intra-block) interior face pairs */
  
  /* work around nvcc assertion failure */
  
  
  /* fluxes for single-sided boundary face pairs */
  
  __syncthreads();
  
  /* zero out unused faces to placate the GTX280's NaN finickiness */
  for (short unsigned int zero_face_nr = BLOCK_FACE; zero_face_nr < data.header.zero_facepairs_count; zero_face_nr += CONCURRENT_FACES)
    ;
  
}
}
