// David S. Noble, Jr.
// Matrix Multiply

#ifndef __MATRIX_UTILS_H__
#define __MATRIX_UTILS_H__

double *allocate_matrix(int rows, int cols);
void deallocate_matrix(double *mat);
double *random_matrix(int rows, int cols);
double *random_matrix_bin(int rows, int cols);
double *ones_matrix(int rows, int cols);
double *zeros_matrix(int rows, int cols);
double *identity_matrix(int rows, int cols);
double *lowerTri_matrix(int rows, int cols);
void copy_block(int procGridX, int procGridY, int rank, int n, int m,
	double *mat, double *dest);
void reorder_matrix(int procGridX, int procGridY, int n, int m,
	double *src, double *dest);
void distribute_matrix(int procGridX, int procGridY, int n, int m,
	double *mat, double *block, int rank);
void verify_element(double a, double b);
void verify_matrix(int m, int n, double *A, double *B);

#endif
