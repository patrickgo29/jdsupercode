// David S. Noble, Jr.
// Matrix Multiply

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <omp.h>

#include "local_mm.h"

void local_mm(const int m, const int n, const int k,
	const double alpha,
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta,
	double *C, const int ldc)
{

	//Verify the sizes of lda, ladb, and ldc
	assert(lda >= m);
	assert(ldb >= k);
	assert(ldc >= m);

	int row, col;
	//Naive implementation
	//Iterate over the columns of C
	for (col = 0; col < n; col++) {

		//Iterate over the rows of C
		for (row = 0; row < m; row++) {

			int k_iter;
			double dotprod = 0.0;

			//Iterate over column of A, row of B
			for (k_iter = 0; k_iter < k; k_iter++) {
				int a_index, b_index;
				a_index = (k_iter * lda) + row;
				b_index = (col * ldb) + k_iter;
				dotprod += A[a_index] * B[b_index]; 
			}

			int c_index = (col * ldc) + row;
			C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
		}
	} 
}

void local_mm_openmp(const int m, const int n, const int k,
	const double alpha,
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta,
	double *C, const int ldc)
{

	//Verify the sizes of lda, ladb, and ldc
	assert(lda >= m);
	assert(ldb >= k);
	assert(ldc >= m);

	//OpenMP implementation
	//Iterate over the columns of C
	int col;
	#pragma omp parallel for
	for (col = 0; col < n; col++) {
		//Iterate over the rows of C
		int row;
		for (row = 0; row < m; row++) {
			int k_iter;
			double dotprod = 0.0;
			//Iterate over column of A, row of B
			for (k_iter = 0; k_iter < k; k_iter++) {
				int a_index = (k_iter * lda) + row; 
				int b_index = (col * ldb) + k_iter;
				dotprod += A[a_index] * B[b_index]; 
			}
			int c_index = (col * ldc) + row;
			C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
		}
	}
}

void local_mm_mkl(const int m, const int n, const int k,
	const double alpha,
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta,
	double *C, const int ldc)
{

	//Verify the sizes of lda, ladb, and ldc 
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
	double *C, const int ldc)
{
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
	int type)
{
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
