// David S. Noble, Jr.
// Matrix Multiply

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "comm_args.h"

//Test the multiplication of two matrices of all ones
void random_multiply(mat_mul_specs * mms);

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

  if (rank == 0) {
    random_multiply(mms);
  }

  MPI_Finalize();
  return 0;
}

void random_multiply(mat_mul_specs * mms) {
  double *A, *B, *C;
  double t_start, t_elapsed;

  //Allocate matrices
  A = random_matrix(mms->m, mms->k);
  B = random_matrix(mms->k, mms->n);
  C = random_matrix(mms->m, mms->n);

  t_start = MPI_Wtime();

  //perform several Matric Mulitplies back-to-back
  int iter;
  for (iter = 0; iter < mms->trials; iter++) {
    //C = (1.0/k)*(A*B) + 0.0*C
    local_mms(mms, 1.0, A, mms->m, B, mms->k, 1.0, C, mms->m);
  }

  t_elapsed = MPI_Wtime() - t_start;

  //deallocate memory
  deallocate_matrix(A);
  deallocate_matrix(B);
  deallocate_matrix(C);

  if(mms->type == NAIVE)
    printf("naive, ");
  else if(mms->type == OPENMP)
    printf("openmp, ");
  else if(mms->type == MKL)
    printf("mkl, ", mms->threads);
  printf("%d, %d, %d, %d, %d, %d, %d, %lf\n", mms->threads, mms->l1, mms->l2, mms->l3, mms->m, mms->n, mms->k, t_elapsed / mms->trials);
}
