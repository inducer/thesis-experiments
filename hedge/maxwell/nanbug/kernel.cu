extern "C" {
texture<float, 1, cudaReadModeElementType> fluxes_on_faces_tex;
texture<float, 1, cudaReadModeElementType> inverse_jacobians_tex;

#define DIMENSIONS 3
#define DOFS_PER_EL 35
#define FACES_PER_EL 4
#define DOFS_PER_FACE 15
#define FACE_DOFS_PER_EL (DOFS_PER_FACE*FACES_PER_EL)

#define CHUNK_DOF threadIdx.x
#define PAR_MB_NR threadIdx.y

#define MB_CHUNK blockIdx.x
#define MACROBLOCK_NR blockIdx.y

#define CHUNK_DOF_COUNT 16
#define MB_CHUNK_COUNT 9
#define MB_DOF_COUNT 144
#define MB_FACEDOF_COUNT 240
#define MB_EL_COUNT 4
#define PAR_MB_COUNT 31
#define SEQ_MB_COUNT 8

#define THREAD_NUM (CHUNK_DOF+PAR_MB_NR*CHUNK_DOF_COUNT)
#define COALESCING_THREAD_COUNT (PAR_MB_COUNT*CHUNK_DOF_COUNT)

#define MB_DOF_BASE (MB_CHUNK*CHUNK_DOF_COUNT)
#define MB_DOF (MB_DOF_BASE+CHUNK_DOF)
#define GLOBAL_MB_NR_BASE (MACROBLOCK_NR*PAR_MB_COUNT*SEQ_MB_COUNT)

#define LIFTMAT_COLUMNS 61
#define LIFTMAT_CHUNK_FLOATS 976
#define LIFTMAT_CHUNK_BYTES (LIFTMAT_CHUNK_FLOATS*4)

__shared__ float smem_lift_mat[LIFTMAT_CHUNK_FLOATS];
__shared__ float dof_buffer[PAR_MB_COUNT][CHUNK_DOF_COUNT];
__shared__ short unsigned int chunk_start_el;
__shared__ short unsigned int chunk_stop_el;
__shared__ short unsigned int chunk_el_count;

__constant__ short unsigned int chunk_start_el_lookup[MB_CHUNK_COUNT]
  = { 0, 0, 0, 1, 1, 2, 2, 3, 3 };
__constant__ short unsigned int chunk_stop_el_lookup[MB_CHUNK_COUNT]
  = { 1, 1, 2, 2, 3, 3, 4, 4, 4 };
__global__ void apply_lift_mat(float *flux, unsigned char *gmem_lift_mat, float *debugbuf)
{
  /* calculate responsibility data */
  unsigned char dof_el = MB_DOF/DOFS_PER_EL;
  
  if (THREAD_NUM==0)
  {
    chunk_start_el = chunk_start_el_lookup[MB_CHUNK];
    chunk_stop_el = chunk_stop_el_lookup[MB_CHUNK];
    chunk_el_count = chunk_stop_el-chunk_start_el;
  }
  __syncthreads();
  
  /* load lift mat chunk */
  {
    unsigned int const *load_base = (unsigned int *) (gmem_lift_mat + MB_CHUNK*LIFTMAT_CHUNK_BYTES);
    for (unsigned word_nr = THREAD_NUM; word_nr*sizeof(int) < (LIFTMAT_CHUNK_BYTES); word_nr += COALESCING_THREAD_COUNT)
      ((unsigned int *) (smem_lift_mat))[word_nr] = load_base[word_nr];
  }
  
  if (chunk_el_count == 1)
    for (unsigned short seq_mb_number = 0; seq_mb_number < SEQ_MB_COUNT; ++seq_mb_number)
    {
      unsigned int global_mb_nr = GLOBAL_MB_NR_BASE + seq_mb_number*PAR_MB_COUNT + PAR_MB_NR;
      unsigned int global_mb_dof_base = global_mb_nr*MB_DOF_COUNT;
      unsigned int global_mb_facedof_base = global_mb_nr*MB_FACEDOF_COUNT;
      
      float result = 0;
      
      dof_buffer[PAR_MB_NR][CHUNK_DOF] = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+(chunk_start_el)*FACE_DOFS_PER_EL+0+CHUNK_DOF);
      __syncthreads();
      
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 0]*dof_buffer[PAR_MB_NR][0];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 1]*dof_buffer[PAR_MB_NR][1];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 2]*dof_buffer[PAR_MB_NR][2];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 3]*dof_buffer[PAR_MB_NR][3];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 4]*dof_buffer[PAR_MB_NR][4];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 5]*dof_buffer[PAR_MB_NR][5];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 6]*dof_buffer[PAR_MB_NR][6];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 7]*dof_buffer[PAR_MB_NR][7];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 8]*dof_buffer[PAR_MB_NR][8];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 9]*dof_buffer[PAR_MB_NR][9];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 10]*dof_buffer[PAR_MB_NR][10];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 11]*dof_buffer[PAR_MB_NR][11];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 12]*dof_buffer[PAR_MB_NR][12];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 13]*dof_buffer[PAR_MB_NR][13];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 14]*dof_buffer[PAR_MB_NR][14];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 15]*dof_buffer[PAR_MB_NR][15];
      
      dof_buffer[PAR_MB_NR][CHUNK_DOF] = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+(chunk_start_el)*FACE_DOFS_PER_EL+16+CHUNK_DOF);
      __syncthreads();
      
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 16]*dof_buffer[PAR_MB_NR][0];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 17]*dof_buffer[PAR_MB_NR][1];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 18]*dof_buffer[PAR_MB_NR][2];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 19]*dof_buffer[PAR_MB_NR][3];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 20]*dof_buffer[PAR_MB_NR][4];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 21]*dof_buffer[PAR_MB_NR][5];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 22]*dof_buffer[PAR_MB_NR][6];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 23]*dof_buffer[PAR_MB_NR][7];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 24]*dof_buffer[PAR_MB_NR][8];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 25]*dof_buffer[PAR_MB_NR][9];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 26]*dof_buffer[PAR_MB_NR][10];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 27]*dof_buffer[PAR_MB_NR][11];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 28]*dof_buffer[PAR_MB_NR][12];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 29]*dof_buffer[PAR_MB_NR][13];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 30]*dof_buffer[PAR_MB_NR][14];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 31]*dof_buffer[PAR_MB_NR][15];
      
      dof_buffer[PAR_MB_NR][CHUNK_DOF] = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+(chunk_start_el)*FACE_DOFS_PER_EL+32+CHUNK_DOF);
      __syncthreads();
      
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 32]*dof_buffer[PAR_MB_NR][0];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 33]*dof_buffer[PAR_MB_NR][1];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 34]*dof_buffer[PAR_MB_NR][2];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 35]*dof_buffer[PAR_MB_NR][3];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 36]*dof_buffer[PAR_MB_NR][4];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 37]*dof_buffer[PAR_MB_NR][5];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 38]*dof_buffer[PAR_MB_NR][6];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 39]*dof_buffer[PAR_MB_NR][7];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 40]*dof_buffer[PAR_MB_NR][8];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 41]*dof_buffer[PAR_MB_NR][9];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 42]*dof_buffer[PAR_MB_NR][10];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 43]*dof_buffer[PAR_MB_NR][11];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 44]*dof_buffer[PAR_MB_NR][12];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 45]*dof_buffer[PAR_MB_NR][13];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 46]*dof_buffer[PAR_MB_NR][14];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 47]*dof_buffer[PAR_MB_NR][15];
      
      dof_buffer[PAR_MB_NR][CHUNK_DOF] = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+(chunk_start_el)*FACE_DOFS_PER_EL+48+CHUNK_DOF);
      __syncthreads();
      
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 48]*dof_buffer[PAR_MB_NR][0];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 49]*dof_buffer[PAR_MB_NR][1];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 50]*dof_buffer[PAR_MB_NR][2];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 51]*dof_buffer[PAR_MB_NR][3];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 52]*dof_buffer[PAR_MB_NR][4];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 53]*dof_buffer[PAR_MB_NR][5];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 54]*dof_buffer[PAR_MB_NR][6];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 55]*dof_buffer[PAR_MB_NR][7];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 56]*dof_buffer[PAR_MB_NR][8];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 57]*dof_buffer[PAR_MB_NR][9];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 58]*dof_buffer[PAR_MB_NR][10];
      result += smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 59]*dof_buffer[PAR_MB_NR][11];
      
      if (MB_DOF < DOFS_PER_EL*MB_EL_COUNT)
        flux[global_mb_dof_base+MB_DOF] = result*tex1Dfetch(inverse_jacobians_tex,global_mb_nr*MB_EL_COUNT+dof_el);
    }
  else
    if (chunk_el_count == 2)
      for (unsigned short seq_mb_number = 0; seq_mb_number < SEQ_MB_COUNT; ++seq_mb_number)
      {
        unsigned int global_mb_nr = GLOBAL_MB_NR_BASE + seq_mb_number*PAR_MB_COUNT + PAR_MB_NR;
        unsigned int global_mb_dof_base = global_mb_nr*MB_DOF_COUNT;
        unsigned int global_mb_facedof_base = global_mb_nr*MB_FACEDOF_COUNT;
        
        float result = 0;
        
        float fof;
        float lm;
        float prev_res;
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+0);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 0];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+1);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 1];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+2);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 2];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+3);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 3];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+4);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 4];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+5);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 5];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+6);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 6];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+7);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 7];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+8);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 8];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+9);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 9];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+10);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 10];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+11);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 11];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+12);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 12];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+13);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 13];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+14);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 14];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+15);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 15];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+16);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 16];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+17);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 17];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+18);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 18];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+19);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 19];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+20);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 20];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+21);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 21];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+22);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 22];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+23);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 23];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+24);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 24];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+25);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 25];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+26);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 26];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+27);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 27];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+28);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 28];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+29);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 29];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+30);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 30];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+31);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 31];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+32);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 32];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+33);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 33];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+34);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 34];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+35);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 35];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+36);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 36];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+37);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 37];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+38);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 38];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+39);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 39];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+40);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 40];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+41);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 41];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+42);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 42];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+43);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 43];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+44);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 44];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+45);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 45];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+46);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 46];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+47);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 47];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+48);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 48];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+49);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 49];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+50);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 50];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+51);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 51];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+52);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 52];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+53);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 53];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+54);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 54];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+55);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 55];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+56);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 56];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+57);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 57];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+58);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 58];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        prev_res = result;
        fof = tex1Dfetch(fluxes_on_faces_tex, global_mb_facedof_base+dof_el*FACE_DOFS_PER_EL+59);
        lm = smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + 59];
        result += fof*lm;
        if (isnan(result))
        {
          debugbuf[MB_DOF*7] = (global_mb_dof_base+MB_DOF);
          debugbuf[MB_DOF*7+1] = fof;
          debugbuf[MB_DOF*7+2] = lm;
          debugbuf[MB_DOF*7+3] = result;
          debugbuf[MB_DOF*7+4] = prev_res;
          debugbuf[MB_DOF*7+5] = fof*lm;
          debugbuf[MB_DOF*7+6] = result+fof*lm;
          goto done;
        }
        
        done:
        if (MB_DOF < DOFS_PER_EL*MB_EL_COUNT)
          flux[global_mb_dof_base+MB_DOF] = result*tex1Dfetch(inverse_jacobians_tex,global_mb_nr*MB_EL_COUNT+dof_el);
      }
}
}
