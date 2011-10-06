/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <omp.h>

#include "local_mm.h"

/**
 *
 *  Local Matrix Multiply
 *   Computes C = alpha * A * B + beta * C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 *  alpha and beta are double-precision scalars
 *
 *  A, B, and C are matrices of double-precision elements
 *  stored in column-major format 
 *
 *  The output is stored in C
 *  A and B are not modified during computation
 *
 *
 *  m - number of rows of matrix A and rows of C
 *  n - number of columns of matrix B and columns of C
 *  k - number of columns of matrix A and rows of B
 * 
 *  lda, ldb, and ldc specifies the size of the first dimension of the matrices
 *
 **/

void local_mm(const int m, const int n, const int k,
                    const double alpha,
                    const double *A, const int lda,
                    const double *B, const int ldb,
                    const double beta,
                    double *C, const int ldc){

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

  int row, col;
  //Naive implementation
  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */
}

void local_mm_openmp(const int m, const int n, const int k,
                    const double alpha,
                    const double *A, const int lda,
                    const double *B, const int ldb,
                    const double beta,
                    double *C, const int ldc){
  
  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

  //OpenMP implementation
  /* Iterate over the columns of C */
  int col;
  #pragma omp parallel for
  for (col = 0; col < n; col++) {
    /* Iterate over the rows of C */
    int row;
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */
}

void local_mm_mkl(const int m, const int n, const int k,
                    const double alpha,
                    const double *A, const int lda,
                    const double *B, const int ldb,
                    const double beta,
                    double *C, const int ldc){

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

  //MKL implementation
  char normal = 'N';
  dgemm(&normal, &normal, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void local_mms(mat_mul_specs * mms, 
		const double alpha, 
		const double *A, const int lda,
		const double *B, const int ldb,
		const double beta, 
		double *C, const int ldc){
	
	switch(mms->type){
		case NAIVE :
			local_mm(mms->m, mms->n, mms->k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		case OPENMP :
			omp_set_num_threads(mms->threads);
			local_mm_openmp(mms->m, mms->n, mms->k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		case MKL :
			omp_set_num_threads(mms->threads);
			local_mm_mkl(mms->m, mms->n, mms->k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		default :
			printf("The intended multiplication type is not recognized!\n");
			abort();
	}
}

void local_mm_typed(const int m, const int n, const int k,
                    const double alpha,
                    const double *A, const int lda,
                    const double *B, const int ldb,
                    const double beta,
                    double *C, const int ldc,
                    int type){
	
	switch(type){
		case NAIVE :
			local_mm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		case OPENMP :
			local_mm_openmp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		case MKL :
			local_mm_mkl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		default :
			printf("The intended multiplication type is not recognized!\n");
			abort();
	}
}
