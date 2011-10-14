// David S. Noble, Jr.
// Matrix Multiply

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

#include "matrix_utils.h"
#include "comm_args.h"
#include "local_mm.h"
#include "summa.h"

#define true 1
#define false 0
#define bool char

#define EPS 0.0001
#define exit_on_fail(passed) (passed)

bool verify_matrix_bool(int m, int n, double *A, double *B)
{

	int row, col;
	for (col = 0; col < n; col++) {
		for (row = 0; row < m; row++) {
			int index = (col * m) + row;
			double a = A[index];
			double b = B[index];

			if (a < b - EPS) {
				return false;
			}
			if (a > b + EPS) {
				return false;
			}
		}
	}

	return true;

}


bool random_matrix_test(int m, int n, int k, int px, int py, 
	int panel_size, int type)
{

	int passed_test = 0, group_passed = 0;
	int num_procs = px * py;
	int rank = 0;
	double *A, *B, *C, *CC, *A_block, *B_block, *C_block, *CC_block;

	A = NULL;
	B = NULL;
	C = NULL;
	CC = NULL;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		A = random_matrix(m, k);
		B = random_matrix(k, n);
		C = zeros_matrix(m, n);
		CC = zeros_matrix(m, n);
		local_mm_typed(m, n, k, 1.0, A, m, B, k, 0.0, CC, m, type);
	}

	A_block = malloc(sizeof(double) * (m * k) / num_procs);
	assert(A_block);

	B_block = malloc(sizeof(double) * (k * n) / num_procs);
	assert(B_block);

	C_block = malloc(sizeof(double) * (m * n) / num_procs);
	assert(C_block);

	CC_block = malloc(sizeof(double) * (m * n) / num_procs);
	assert(CC_block);

	distribute_matrix(px, py, m, k, A, A_block, rank);
	distribute_matrix(px, py, k, n, B, B_block, rank);
	distribute_matrix(px, py, m, n, C, C_block, rank);
	distribute_matrix(px, py, m, n, CC, CC_block, rank);

	if (rank == 0) {
		deallocate_matrix(A);
		deallocate_matrix(B);
		deallocate_matrix(C);
		deallocate_matrix(CC);
	}

	summa_typed(m, n, k, A_block, B_block, C_block, px, py, 1, type);

	if (verify_matrix_bool(m / px, n / py, C_block, CC_block) == false){
		passed_test = 1;
	}

	free(A_block);
	free(B_block);
	free(C_block);
	free(CC_block);

	MPI_Reduce(&passed_test, &group_passed, 1, MPI_INT, MPI_SUM, 0,
	MPI_COMM_WORLD);
	MPI_Bcast(&group_passed, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0 && group_passed == 0) {
		printf("random_matrix_test m=%d n=%d k=%d px=%d py=%d pb=%d\
			............PASSED\n", m, n, k, px, py, panel_size);
	}

	if (rank == 0 && group_passed != 0) {
		printf("random_matrix_test m=%d n=%d k=%d px=%d py=%d pb=%d\
			............FAILED\n", m, n, k, px, py, panel_size);
	}

	if (group_passed == 0) {
		return true;
	} else {
		return false;
	}
}

int main(int argc, char *argv[]) {
	int rank = 0;
	int np = 0;
	char hostname[MPI_MAX_PROCESSOR_NAME + 1];
	int namelen = 0;

	MPI_Init(&argc, &argv); /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */
	MPI_Comm_size(MPI_COMM_WORLD, &np); /* Get number of processes */
	MPI_Get_processor_name(hostname, &namelen); /* Get hostname of node */
	printf("[Using Host:%s -- Rank %d out of %d]\n", hostname, rank, np);

	/* These tests use 16 processes */
	if (np != 16) {
		printf("Error: np=%d. Please use 16 processes\n", np);
	}

	if(rank == 0){
		printf("\n");
	}

	int type;
	for(type = 0; type <= 2; type++){
		
		if(rank == 0){
			if(type == NAIVE)
				printf("testing the naive local_mm\n");
			if(type == OPENMP)
				printf("testing the OpenMP local_mm\n");
			if(type == MKL)
				printf("testing the MKL local_mm\n");
		}
		
		exit_on_fail( random_matrix_test(16, 16, 16, 4, 4, 1, type));
		exit_on_fail( random_matrix_test(32, 32, 32, 4, 4, 1, type));
		exit_on_fail( random_matrix_test(128, 128, 128, 4, 4, 1, type));

		exit_on_fail( random_matrix_test(128, 32, 128, 4, 4, 1, type));
		exit_on_fail( random_matrix_test(64, 32, 128, 4, 4, 1, type));

		exit_on_fail( random_matrix_test(128, 128, 128, 8, 2, 1, type));
		exit_on_fail( random_matrix_test(128, 128, 128, 2, 8, 1, type));
		exit_on_fail( random_matrix_test(128, 128, 128, 1, 16, 1, type));
		exit_on_fail( random_matrix_test(128, 128, 128, 16, 1, 1, type));

		printf("\n");
	}
	
	MPI_Finalize();
	
	return 0;
}
