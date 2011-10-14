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
void identity_test(int n, int type) {
	double *A, *B, *C;

	printf("identity_test n=%d............", n);

	//Allocate matrices
	A = random_matrix(n, n);
	B = identity_matrix(n, n);
	C = zeros_matrix(n, n);

	//C = 1.0*(A*B) + 0.0*C
	local_mm_typed(n, n, n, 1.0, A, n, B, n, 5.0, C, n, type);

	//Verfiy the results
	verify_matrix(n, n, A, C);

	//Backwards C = 1.0*(B*A) + 0.0*C
	local_mm_typed(n, n, n, 1.0, B, n, A, n, 0.0, C, n, type);

	//Verfiy the results
	verify_matrix(n, n, A, C);

	//deallocate memory
	deallocate_matrix(A);
	deallocate_matrix(B);
	deallocate_matrix(C);

	printf("passed\n");
}

//Test the multiplication of two matrices of all ones
void ones_test(int m, int n, int k, int type) {
	double *A, *B, *C, *C_ones, *C_zeros;

	printf("ones_test m=%d n=%d k=%d............", m, n, k);

	//Allocate matrices
	A = ones_matrix(m, k);
	B = ones_matrix(k, n);
	C = ones_matrix(m, n);

	C_ones = ones_matrix(m, n);
	C_zeros = zeros_matrix(m, n);

	//C = (1.0/k)*(A*B) + 0.0*C
	local_mm_typed(m, n, k, (1.0 / k), A, m, B, k, 0.0, C, m, type);

	//Verfiy the results
	verify_matrix(m, n, C, C_ones);

	//C = (1.0/k)*(A*B) + -1.0*C
	local_mm_typed(m, n, k, (1.0 / k), A, m, B, k, -1.0, C, m, type);

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
void lower_triangular_test(int n, int type) {

	int i;
	double *A, *B, *C;

	printf("lower_triangular_test n=%d ............", n);

	//Allocate matrices
	A = lowerTri_matrix(n, n);
	B = ones_matrix(n, 1);
	C = ones_matrix(n, 1);

	//C = 1.0*(A*B) + 0.0*C
	local_mm_typed(n, 1, n, 1.0, A, n, B, n, 0.0, C, n, type);

	//Loops over every element in C
	//The elements of C should be [1 2 3 4...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) (i + 1.0));

	//C = 0.0*(A*B) + 1.0*C
	local_mm_typed(n, 1, n, 0.0, A, n, B, n, 1.0, C, n, type);

	//Loops over every element in C
	//The elements of C should be [1 2 3 4...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) (i + 1.0));

	//C = 1.0*(A*B) + 1.0*C
	local_mm_typed(n, 1, n, 3.0, A, n, B, n, 1.0, C, n, type);

	//Loops over every element in C
	//The elements of C should be [4 8 16 32...]
	for (i = 0; i < n; i++)
		assert(C[i] == (double) ((i + 1) * 4.0));

	//C = 0.0*(A*B) + 0.0*C
	local_mm_typed(n, 1, n, 0.0, A, n, B, n, 0.0, C, n, type);

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
	int type;
	for(type = 0; type <= 2; type++){
		switch(type)
		{
			case NAIVE:
				printf("Testing the naive local_mm\n");
				break;
			case OPENMP:
				printf("Testing the OpenMP local_mm\n");
				break;
			case MKL:
				printf("Testing the MKL local_mm\n");
				break;
		}
		identity_test(16, type);
		identity_test(37, type);
		identity_test(512, type);
		ones_test(32, 32, 32, type);
		ones_test(61, 128, 123, type);
		lower_triangular_test(8, type);
		lower_triangular_test(92, type);
		lower_triangular_test(128, type);
		printf("\n");
	}

	return 0;
}

