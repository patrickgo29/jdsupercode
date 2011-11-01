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

/* SSE double precision matrix multiply routine
 * Uses block size of 4 (the most that can fit on 16 registers)
 *
 */

void 
basic_dgemm (const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
    int i,j,k;
    int rs = REGISTER_SIZE;
    
    // init __m128d datatypes 
    __m128d A1,A2,B1,B2,B3,B4,
            C1,C2,C3,C4,C5,C6,C7,C8;

    for(j=0; j<N; j+=rs) 
    {
        for(i=0; i<M; i+=rs)
        {
           C += i+j*lda; // update C to point to correct block
           C1 = _mm_load_pd(C); C+=2;
           C2 = _mm_load_pd(C); C+=lda-2;
           C3 = _mm_load_pd(C); C+=2;
           C4 = _mm_load_pd(C); C+=lda-2;
           C5 = _mm_load_pd(C); C+=2;
           C6 = _mm_load_pd(C); C+=lda-2;
           C7 = _mm_load_pd(C); C+=2;
           C8 = _mm_load_pd(C);
           C += (-3*lda)-2; //reset C

            for(k=0; k<K; k+=1)
            {
                //load {A1,A2,B1,B2,B3,B4}
                A1 = _mm_load_pd(A+i+k*lda);
                A2 = _mm_load_pd(A+i+k*lda+2);
                B1 = _mm_load1_pd(B+j+k*lda);
                B2 = _mm_load1_pd(B+j+k*lda+1);
                B3 = _mm_load1_pd(B+j+k*lda+2);
                B4 = _mm_load1_pd(B+j+k*lda+3);

                //update Ci's
                C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
                C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
                C3 = _mm_add_pd(_mm_mul_pd(A1,B2),C3);
                C4 = _mm_add_pd(_mm_mul_pd(A2,B2),C4);
                C5 = _mm_add_pd(_mm_mul_pd(A1,B3),C5);
                C6 = _mm_add_pd(_mm_mul_pd(A2,B3),C6);
                C7 = _mm_add_pd(_mm_mul_pd(A1,B4),C7);
                C8 = _mm_add_pd(_mm_mul_pd(A2,B4),C8);
            }
            
            // move results to main memory
            _mm_store_sd(C,C1); C+=2;
            _mm_store_sd(C,C2); C+=lda-2;
            _mm_store_sd(C,C3); C+=2;
            _mm_store_sd(C,C4); C+=lda-2;
            _mm_store_sd(C,C5); C+=2;
            _mm_store_sd(C,C6); C+=lda-2;
            _mm_store_sd(C,C7); C+=2;
            _mm_store_sd(C,C8);
            C += (-3*lda)-2;

            // restore C for next iteration
            C -= i+j*lda;
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
  posix_memalign((void**)&A_copy,16,copysize);
  posix_memalign((void**)&B_copy,16,copysize);
  posix_memalign((void**)&C_copy,16,copysize);

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
