// David S. Noble, Jr.
// Matrix Multiply

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "matrix_utils.h"
#include "local_mm.h"

//Verify that a matrix times the identity is itself
void identity_test(int n, mat_mul_specs * mms) {
	double *A, *B, *C;

	printf("identity_test n=%d............", n);

	//Allocate matrices
	A = random_matrix(n, n);
	B = identity_matrix(n, n);
	C = zeros_matrix(n, n);

	//C = 1.0*(A*B) + 0.0*C
	local_mm_mms(n, n, n, 1.0, A, n, B, n, 5.0, C, n, mms);

	//Verfiy the results
	verify_matrix(n, n, A, C);

	//Backwards C = 1.0*(B*A) + 0.0*C
	local_mm_mms(n, n, n, 1.0, B, n, A, n, 0.0, C, n, mms);

	//Verfiy the results
	verify_matrix(n, n, A, C);

	//deallocate memory
	deallocate_matrix(A);
	deallocate_matrix(B);
	deallocate_matrix(C);

	printf("passed\n");
}

//Test the multiplication of two matrices of all ones
void ones_test(int m, int n, int k, mat_mul_specs * mms) {
	double *A, *B, *C, *C_ones, *C_zeros;

	printf("ones_test m=%d n=%d k=%d............", m, n, k);

	//Allocate matrices
	A = ones_matrix(m, k);
	B = ones_matrix(k, n);
	C = ones_matrix(m, n);

	C_ones = ones_matrix(m, n);
	C_zeros = zeros_matrix(m, n);

	//C = (1.0/k)*(A*B) + 0.0*C
	local_mm_mms(m, n, k, (1.0 / k), A, m, B, k, 0.0, C, m, mms);

	//Verfiy the results
	verify_matrix(m, n, C, C_ones);

	//C = (1.0/k)*(A*B) + -1.0*C
	local_mm_mms(m, n, k, (1.0 / k), A, m, B, k, -1.0, C, m, mms);

	//Verfiy the results
	verify_matrix(m, n, C, C_zeros);

	//deallocate memory
	deallocate_matrix(A);
	deallocate_matrix(B);
	deallocate_matrix(C);

	deallocate_matrix(C_ones);
	deallocate_matrix(C_zeros);

	printf("passed\n");
}

//Test the multiplication of a lower triangular matrix
void lower_triangular_test(int n, mat_mul_specs * mms) {

	int i;
	double *A, *B, *C;

	printf("lower_triangular_test n=%d ............", n);

	//Allocate matrices
	A = lowerTri_matrix(n, n);
	B = ones_matrix(n, 1);
	C = ones_matrix(n, 1);

	//C = 1.0*(A*B) + 0.0*C
	local_mm_mms(n, 1, n, 1.0, A, n, B, n, 0.0, C, n, mms);

	//Loops over every element in C
	//The elements of C should be [1 2 3 4...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) (i + 1.0));

	//C = 0.0*(A*B) + 1.0*C
	local_mm_mms(n, 1, n, 0.0, A, n, B, n, 1.0, C, n, mms);

	//Loops over every element in C
	//The elements of C should be [1 2 3 4...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) (i + 1.0));

	//C = 1.0*(A*B) + 1.0*C
	local_mm_mms(n, 1, n, 3.0, A, n, B, n, 1.0, C, n, mms);

	//Loops over every element in C
	//The elements of C should be [4 8 16 32...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) ((i + 1) * 4.0));

	//C = 0.0*(A*B) + 0.0*C
	local_mm_mms(n, 1, n, 0.0, A, n, B, n, 0.0, C, n, mms);

	//Loops over every element in C
	//The elements of C should be 0
	for (i = 0; i < n; i++)
		assert(C[i] == 0.0);

	//deallocate memory
	deallocate_matrix(A);
	deallocate_matrix(B);
	deallocate_matrix(C);

	printf("passed\n");
}

int main() {
	mat_mul_specs mms;
	int type;
	for(type = 0; type <= 3; type++){
		switch(type)
		{
			case 0:
				printf("Testing the naive local_mm\n");
				mms.type = NAIVE;
				break;
			case 1:
				printf("Testing the OpenMP local_mm\n");
				mms.type = OPENMP;
				mms.cbl = 0;
				mms.cop = 0;
				mms.threads = 12;
				break;
			case 2:
				printf("Testing the MKL local_mm\n");
				mms.type = MKL;
				mms.threads = 12;
				break;
			case 3:
				printf("Testing the OpenMP cache blocking local_mm\n");
				mms.type = OPENMP;
				mms.cbl = 1;
				mms.cop = 0;
				mms.threads = 12;
				mms.bm = 2;
				mms.bn = 2;
				mms.bk = 2;
				break;
			case 4:
				printf("Testing the OpenMP cache blocking and copy optimization local_mm\n");
				mms.type = OPENMP;
				mms.cbl = 1;
				mms.cop = 2;
				mms.threads = 12;
				mms.bm = 2;
				mms.bn = 2;
				mms.bk = 2;
				break;
	
		}
		identity_test(16, &mms);
		identity_test(37, &mms);
		identity_test(512, &mms);
		ones_test(32, 32, 32, &mms);
		ones_test(61, 128, 123, &mms);
		lower_triangular_test(8, &mms);
		lower_triangular_test(92, &mms);
		lower_triangular_test(128, &mms);
		printf("\n");
	}

	return 0;
}

