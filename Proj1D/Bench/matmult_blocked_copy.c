/**
 *  \file matmult.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Aparna Chandramowlishwaran <aparna@gatech...>, Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
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
#define REG_SIZE 4
#endif

#define MIN(a, b) (a < b) ? a : b

void 
basic_dgemm (const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
  int K_iter = MIN(BLOCK_SIZE,K);
      int i, j, k;
      for (i = 0; i < M; i+=REG_SIZE) {
        for (j = 0; j < N; j+=REG_SIZE) {
          for (k = 0; k < K_iter; k+=REG_SIZE) {
              sse_kernel_four_trans(K_iter,
                                       lda,
                                       A+k+i*K_iter,
                                       B+k+j*K_iter,
                                       C+i+j*lda);
      }
    } 
  }
}

void 
dgemm_copy (const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
  int i, j, k;

  double *A_temp = (double*)memalign(64,BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  double *B_temp = (double*)memalign(64,BLOCK_SIZE*BLOCK_SIZE*sizeof(double));

  /* Copy matrix A into cache */  
  for (i = 0; i < K; i++) {
    for (j = 0; j < M; j++) {
      A_temp[i + j*K] = A[j + i*lda];
    }
  }

  /* Copy matrix B into cache */
  for (i = 0; i < N; i++) {
    for (j = 0; j < K; j++) {
      B_temp[j + i*K] = B[j + i*lda];
    }
  }
  
  basic_dgemm (lda, M, N, K, A_temp, B_temp, C);

  free(A_temp);
  free(B_temp);
}

void 
matmult (const int lda, const double *A, const double *B, double *C) 
{
  int i;

#pragma omp parallel for shared (lda,A,B,C) private (i)
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
