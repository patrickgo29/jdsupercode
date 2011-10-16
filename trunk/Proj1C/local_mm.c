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

	//Naive implementation
	//Iterate over the columns of C
	for (int col = 0; col < n; col++) {
		//Iterate over the rows of C
		for (int row = 0; row < m; row++) {
			double dotprod = 0.0;
			//Iterate over column of A, row of B
			for (int k_iter = 0; k_iter < k; k_iter++) {
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
	#pragma omp parallel for
	for (int col = 0; col < n; col++) {
		//Iterate over the rows of C
		for (int row = 0; row < m; row++) {
			double dotprod = 0.0;
			//Iterate over column of A, row of B
			for (int k_iter = 0; k_iter < k; k_iter++) {
				int a_index = (k_iter * lda) + row; 
				int b_index = (col * ldb) + k_iter;
				dotprod += A[a_index] * B[b_index]; 
			}
			int c_index = (col * ldc) + row;
			C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
		}
	}
}

void local_mm_openmp_cbl_outer(const int m, const int n, const int k,
	const double alpha,
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta,
	double *C, const int ldc,
	int l1, int l2, int l3)
{

	//Verify the sizes of lda, ldb, and ldc
	assert(lda >= m);
	assert(ldb >= k);
	assert(ldc >= m);

	l3 = l3 / omp_get_num_threads();
	int bk = k / (k * m / l3);
	int bm = bk / (bk * m / l2);
	int bn = n / (bk * n / l1);
	#pragma omp parallel for
	for (int b_k = 0; b_k < k; b_k += bk) {
		for (int b_i = 0; b_i < m; b_i += bm) {
			for (int b_j = 0; b_j < n; b_j += bn){
//////////////////////////////////////////////////////////////////////
				for (int col = b_j; col < n && col < b_j + bn; col++) {
					for (int row = b_i; row < m && row < b_i + bm; row++) {
						double dotprod = 0.0;
						for (int k_iter = 0; k_iter < k; k_iter++) {
							int a_index = (k_iter * lda) + row; 
							int b_index = (col * ldb) + k_iter;
							dotprod += A[a_index] * B[b_index]; 
						}
						int c_index = (col * ldc) + row;
						if(b_k == 0){
							C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
						}else{
							C[c_index] += (alpha * dotprod);
						}
					}
				}
//////////////////////////////////////////////////////////////////////
			}
		}
	}
}

void local_mm_openmp_cbl_cop(const int m, const int n, const int k,
	const double alpha,
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta,
	double *C, const int ldc,
	int l1, int l2, int l3)
{

	//Verify the sizes of lda, ladb, and ldc
	assert(lda >= m);
	assert(ldb >= k);
	assert(ldc >= m);

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

void local_mm_mms(const int m, const int n, const int k, 
	const double alpha, 
	const double *A, const int lda,
	const double *B, const int ldb,
	const double beta, 
	double *C, const int ldc
	mat_mul_specs * mms)
{
	switch(mms->type){
		case NAIVE :
			local_mm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		case OPENMP :
			omp_set_num_threads(mms->threads);
			if(!mms->cbl){
				local_mm_openmp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			}else if(!mms->cop){
				if(mms->inner){
					local_mm_openmp_cbl_inner(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mms->l1, mms->l2, mms->l3);
				}else{
					local_mm_openmp_cbl_outer(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mms->l1, mms->l2, mms->l3);
				}
			}else{
				if(mms->inner){
					local_mm_openmp_cbl_cop_inner(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mms->l1, mms->l2, mms->l3);
				}else{
					local_mm_openmp_cbl_cop_outer(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mms->l1, mms->l2, mms->l3);
				}
			}
			break;
		case MKL :
			omp_set_num_threads(mms->threads);
			local_mm_mkl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;
		default :
			printf("The intended multiplication type is not recognized!\n");
			abort();
	}
}
