/**
 *  \file time_summa.c
 *  \brief code for timing summa()
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "summa.h"
#include "comm_args.h"

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

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */

  //if (rank == 0) {
  //  printf(
  //      "random_matrix_test m=%d n=%d k=%d px=%d py=%d pb=%d iterations=%d.....",
  //      m, n, k, px, py, pb, iterations);

  //}

  /*  Initialize matrix blocks */
  A_block = random_matrix(m / px, k / py);
  assert(A_block);

  B_block = random_matrix(k / px, n / py);
  assert(B_block);

  C_block = random_matrix(m / px, n / py);
  assert(C_block);

  MPI_Barrier( MPI_COMM_WORLD);

  /* 
   *
   * Call SUMMA
   *
   */

  t_start = MPI_Wtime(); /* Start timer */
  for (iter = 0; iter < iterations; iter++) {
    summa_mms(mms, A_block, B_block, C_block);
  } /* iter */

  MPI_Barrier(MPI_COMM_WORLD);
  t_elapsed = MPI_Wtime() - t_start; /* Stop timer */

  if (rank == 0) {
    printf("%d, %d, %d, %d, %d, %d, %lf, %d, %lf, ", mms->m, mms->n, mms->k, mms->x, mms->y, mms->b, t_elapsed, mms->trials, t_elapsed / mms->trials);
    if(mms->type == 0)
      printf("naive, ");
    if(mms->type == 1)
      printf("openmp, ");
    if(mms->type == 2)
      printf("mkl, ");
    printf("%d\n", mms->threads);
  }

  deallocate_matrix(A_block);
  deallocate_matrix(B_block);
  deallocate_matrix(C_block);
}

/** Program start */
int main(int argc, char *argv[]) {

  mat_mul_specs * mms = getMatMulSpecs(argc, argv);

  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME + 1];
  int namelen = 0;

  MPI_Init(&argc, &argv); /* starts MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */
  MPI_Comm_size(MPI_COMM_WORLD, &np); /* Get number of processes */
  MPI_Get_processor_name(hostname, &namelen); /* Get hostname of node */
  //printf("Using Host:%s -- Rank %d out of %d\n", hostname, rank, np);

  random_summa(mms);

  MPI_Finalize();
  return 0;
}
