// David S. Noble, Jr.
// Matrix Multiply

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "summa.h"
#include "comm_args.h"

void random_summa(mat_mul_specs * mms);

int main(int argc, char *argv[]) {

  mat_mul_specs * mms = getMatMulSpecs(argc, argv);

  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME + 1];
  int namelen = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Get_processor_name(hostname, &namelen);
  
  random_summa(mms);

  MPI_Finalize();
  return 0;
}

void random_summa(mat_mul_specs * mms) {
  int m, n, k, px, py, pb, iterations;
  m = mms->m;
  n = mms->n;
  k = mms->k;
  px = mms->x;
  py = mms->y;
  pb = mms->b;
  iterations = mms->trials;
  int iter;
  double t_start, t_elapsed;
  int rank = 0;
  double *A_block, *B_block, *C_block;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  A_block = random_matrix(m / px, k / py);
  assert(A_block);

  B_block = random_matrix(k / px, n / py);
  assert(B_block);

  C_block = random_matrix(m / px, n / py);
  assert(C_block);

  MPI_Barrier( MPI_COMM_WORLD);

  t_start = MPI_Wtime();
  for (iter = 0; iter < iterations; iter++) {
    summa_mms(mms, A_block, B_block, C_block);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t_elapsed = MPI_Wtime() - t_start;

  deallocate_matrix(A_block);
  deallocate_matrix(B_block);
  deallocate_matrix(C_block);

  if (rank == 0) {
    if(mms->type == NAIVE)
      printf("naive, ");
    else if(mms->type == OPENMP)
      printf("openmp, ");
    else if(mms->type == MKL)
      printf("mkl, ");
    printf("%d, %d, %d, %d, %d, %d, %lf, %d, %lf\n", mms->m, mms->n, \
		mms->k, mms->x, mms->y, mms->b, t_elapsed, mms->trials, \
		t_elapsed / mms->trials);
  }
}
