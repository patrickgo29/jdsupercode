/**
 *  \file matmult.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Aparna Chandramowlishwaran <aparna@gatech...>, Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>

#include "matmult.h"

/**
 *
 *  Matrix Multiply
 *   Computes C = A * B + C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 **/

#if !defined (BLOCK_SIZE)
#define BLOCK_SIZE 64
#define REGISTER_SIZE 4
#endif

#define MIN(a, b) (a < b) ? a : b


/* basic_dgemm matrix multiply routine. should work for more general matrix sizes
 * requires for dimensions of A,B,C to be a multiple of 4
 *
 * differences from above - got rid of scalar multiplication, all computations are
 *    vectorized
 * does not use blocking, but rather line size - only uses 5 registers now instead of 14
 *    to achieve same effect
 * TODO: implement larger line size with more registers
 *
 */

void basic_dgemm (const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
    int i,j,k;
    __m128d A1,A2,B1,C1,C2;
    int rs = REGISTER_SIZE;

    for (i=0; i<M; i+=rs) {
        for (j=0; j<N; j++) {
            A1 = _mm_load_pd(A+i);
            A2 = _mm_load_pd(A+i+2);
            B1 = _mm_set1_pd(B[j]);
            C1 = _mm_mul_pd(A1,B1);
            C2 = _mm_mul_pd(A2,B1);
            for (k=1; k<K; k++) {
                A1 = _mm_load_pd(A+i+k*lda);
                A2 = _mm_load_pd(A+i+k*lda+2);
                B1 = _mm_set1_pd(B[j+k*lda]);
                C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
                C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
            }
            _mm_store_pd(C+i+j*lda,C1);
            _mm_store_pd(C+i+j*lda+2,C2);
        }
    }
}

void 
dgemm_copy(const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
  int i, j;
  int rs = REGISTER_SIZE;
  int pad, offset, num;
  double *B_temp;
  double *A_copy, *B_copy, *C_copy;
  size_t copysize;
  
  // if lda is a power of two, no alignment is necessary

  if (lda%rs==0) {
    pad = lda;
  } else {
    pad = (lda+rs)-(lda%rs);
  }
    
  // allocate A_copy and C_copy
  // align to 128 bytes, since that is the size of the block that
  // basic_dgemm can do
  copysize = pad*pad*sizeof(double);
  posix_memalign((void**)&A_copy,128,copysize);
  posix_memalign((void**)&B_copy,128,copysize);
  posix_memalign((void**)&C_copy,128,copysize);

  /* Copy matrix A */ 
  for (i=0; i<M; i++) {
    for (j=0; j<K; j++) {
      A_copy[i+j*pad] = A[i+j*lda];
    }
  }
  
  /* Copy matrix B */
  for (i=0; i<K; i++) {
    for (j=0; j<N; j++) {
      B_copy[j+i*pad] = B[i+j*lda]; //transpose B
    }
  }

  /* Copy matrix C */
  for (i=0; i<M; i++) {
    for (i=0; i<N; i++) {
      C_copy[i+j*pad] = C[i+j*lda];
    }
  }
  
  /* Perform call to register dgemm code */
  basic_dgemm (lda, M, N, K, A_copy, B_copy, C_copy);

  /* Copy results back to C and free up memory */
  if (lda%rs!=0) {
      for (i=0; i<M; i++) { 
          for (j=0; j<N; j++) {
              C[i+j*lda] = C_copy[i+j*pad];
          }
      }
  }
  free(A_copy);
  free(B_copy);
  free(C_copy);
}

void 
matmult (const int lda, const double *A, const double *B, double *C) 
{
  int i;

//#pragma omp parallel for shared (lda,A,B,C) private (i)
  for (i = 0; i < lda; i += BLOCK_SIZE) {
    int j;
    for (j = 0; j < lda; j += BLOCK_SIZE) {
      int k;
      for (k = 0; k < lda; k += BLOCK_SIZE) {
        int M = MIN (BLOCK_SIZE, lda-i);
        int N = MIN (BLOCK_SIZE, lda-j);
        int K = MIN (BLOCK_SIZE, lda-k);

        dgemm_copy (lda, M, N, K, A+i+k*lda, B+k+j*lda, C+i+j*lda);
      }
    }
  }  
}
