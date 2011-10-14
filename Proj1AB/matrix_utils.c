// David S. Noble, Jr.
// Matrix Multiply

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "matrix_utils.h"

#define EPSILON 0.0001

//Allocates a matrix
double *allocate_matrix(int rows, int cols) {
	double *mat = NULL;
	mat = malloc(sizeof(double) * rows * cols);
	assert(mat != NULL);
	return (mat);
}

//Deallocates a matrix
void deallocate_matrix(double *mat) {
	free(mat);
}

//Set the elements of the matrix to random values
double *random_matrix(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			mat[index] = round(10.0 * rand() / (RAND_MAX + 1.0));
		}
	}

	return mat;
}

//Set the elements of the matrix to random values
double *random_matrix_bin(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			mat[index] = round(rand() / (RAND_MAX + 1.0));
		}
	}

	return mat;
}

//Sets each element of the matrix to 1
double *ones_matrix(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			mat[index] = 1.0;
		}
	}

	return mat;
}

//Sets each element of the matrix to 0
double *zeros_matrix(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			mat[index] = 0.0;
		}
	}

	return mat;
}

//Sets each element of the diagonal to 1, 0 otherwise
double *identity_matrix(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			if (r == c) {
				mat[index] = 1.0;
			} else {
				mat[index] = 0.0;
			}
		}
	}

	return mat;
}

//Create a lower triangular matrix of 1's
double *lowerTri_matrix(int rows, int cols) {

	int r, c;
	double *mat = allocate_matrix(rows, cols);

	for (c = 0; c < cols; c++) {
		for (r = 0; r < rows; r++) {
			int index = (c * rows) + r;
			if (r >= c) {
				mat[index] = 1.0;
			} else {
				mat[index] = 0.0;
			}
		}
	}

	return mat;
}

//copy a block of the matrix to a buffer
void copy_block(int procGridX, int procGridY, int rank, int n, int m, double *mat, 
	double *dest)
{

	int row, col;
	int block_index = 0;

	int block_rows = n / procGridX;
	int block_cols = m / procGridY;

	int proc_x = rank % procGridX;
	int proc_y = (rank - proc_x) / procGridX;

	assert(n % procGridX == 0);
	assert(m % procGridY == 0);

	for (col = proc_y * block_cols; col < (proc_y + 1) * block_cols; col++) {
		for (row = proc_x * block_rows; row < (proc_x + 1) * block_rows; row++) {
			int mat_index = (col * n) + row;
			dest[block_index] = mat[mat_index];
			block_index++;
		}
	}
}

//Reorders matrix from buffers
void reorder_matrix(int procGridX, int procGridY, int n, int m, double *src, 
	double *dest)
{

	int block;
	int num_blocks = procGridX * procGridY;
	int block_size = m * n / num_blocks;

	assert(n % procGridX == 0);
	assert(m % procGridY == 0);

	for (block = 0; block < num_blocks; block++) {
		copy_block(procGridX, procGridY, block, n, m, src, 
			&(dest[block_size * block]));
	}
}

//Distributes the matrix from the head node across all nodes
void distribute_matrix(int procGridX, int procGridY, int n, int m, double *mat, 
	double *block, int rank)
{

	double *buffer = NULL;

	int num_procs = procGridX * procGridY;
	int block_size = m * n / num_procs;

	if (rank == 0) {
		buffer = malloc(sizeof(double) * m * n);
		assert(buffer != NULL);

		reorder_matrix(procGridX, procGridY, n, m, mat, buffer);
	}

	MPI_Scatter(buffer, block_size, MPI_DOUBLE, block, block_size, MPI_DOUBLE, 
		0, MPI_COMM_WORLD);

	if (rank == 0) {
		free(buffer);
	}
}

//Verify relative equivalence of the elements
void verify_element(double a, double b)
{

	assert(a < (b + EPSILON));
	assert(a > (b - EPSILON));
}

//Verify relative equivalence of the matrices
void verify_matrix(int m, int n, double *A, double *B)
{
	
	int i;

	for (i = 0; i < m * n; i++) {
		verify_element(A[i], B[i]);
	}
}

//Allocate and distribute a matrix across multiple nodes
void allocate_and_distribute(double *mat, double *block, int m, int n, int procGridX, int procGridY, int rank)
{

	int num_procs = procGridX * procGridY;

	block = malloc(sizeof(double) * (m * n) / num_procs);
	assert(block);

	distribute_matrix(procGridX, procGridY, m, n, mat, block, rank);
}
