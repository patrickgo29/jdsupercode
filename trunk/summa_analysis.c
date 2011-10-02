/**
 *  \file unittest_summa.c
 *  \brief unittests for summa()
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "summa.h"

#define true 1
#define false 0
#define bool char

#define EPS 0.0001

/**
 * Function to compute elapsed time in timeval result between two timevals t1 and t2
 * Returns 1 if diff between two times is negative
 */

int timeval_subtract (result, x, y) 
    struct timeval *result, *x, *y;
{
   /* Perform the carry for the later subtraction by updating y. */
   if (x->tv_usec < y->tv_usec) {
      int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
      y->tv_usec -= 1000000 * nsec;
      y->tv_sec += nsec;
   }
 
   if (x->tv_usec - y->tv_usec > 1000000) {
      int nsec = (x->tv_usec - y->tv_usec) / 1000000;
      y->tv_usec += 1000000 * nsec;
      y->tv_sec -= nsec;
   }
    
   /* Compute the time remaining to wait.
    *           tv_usec is certainly positive. */
   result->tv_sec = x->tv_sec - y->tv_sec;
   result->tv_usec = x->tv_usec - y->tv_usec;
                   
   /* Return 1 if result is negative. */
   return x->tv_sec < y->tv_sec;
}

/* Similar to verify_matrix(),
 *  this function verifies that each element of A
 *  matches the corresponding element of B
 *
 *  returns true if A and B are equal
 */
bool verify_matrix_bool(int m, int n, double *A, double *B) {

  /* Loop over every element of A and B */
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
    } /* row */
  }/* col */

  return true;
}


/**
 * Creates random A, B, and C matrices and uses summa() to
 *  calculate the product. Returns time taken to compute
 *  summa
 **/
void random_matrix_test(int m, int n, int k, int px, int py, int panel_size) {

  int num_procs = px * py;
  int rank = 0;
  double *A, *B, *C, *A_block, *B_block, *C_block;
  struct timeval tvbegin, tvend, tvresult;
  double procTime; 
  double totalTime;

  A = NULL;
  B = NULL;
  C = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */

  if (rank == 0) {
    /* Allocate matrices */
    A = random_matrix(m, k);
    B = random_matrix(k, n);
    C = zeros_matrix(m, n);
  }

  /* 
   * Allocate memory for matrix blocks 
   */
  A_block = malloc(sizeof(double) * (m * k) / num_procs);
  assert(A_block);

  B_block = malloc(sizeof(double) * (k * n) / num_procs);
  assert(B_block);

  C_block = malloc(sizeof(double) * (m * n) / num_procs);
  assert(C_block);

  /* Distrute the matrices */
  distribute_matrix(px, py, m, k, A, A_block, rank);
  distribute_matrix(px, py, k, n, B, B_block, rank);
  distribute_matrix(px, py, m, n, C, C_block, rank);

  if (rank == 0) {

    /* 
     * blocks of A, B, C, and CC have been distributed to
     * each of the processes, now we can safely deallocate the 
     * matrices
     */
    deallocate_matrix(A);
    deallocate_matrix(B);
    deallocate_matrix(C);
  }
  /* flush output and synchronize the processes */  
  fflush(stdout);
  sleep(1);
  MPI_Barrier(MPI_COMM_WORLD);

  /*
   *  Calculate time for each process to run SUMMA for its own block
   *
   */
  gettimeofday(&tvbegin, NULL);
  summa(m, n, k, A_block, B_block, C_block, px, py, panel_size);
  gettimeofday(&tvend, NULL);

  fflush(stdout);
  sleep(1);
  MPI_Barrier(MPI_COMM_WORLD);

  /* free resources */
  free(A_block); free(B_block); free(C_block);

  /* compute time for process to complete SUMMA algorithm */
  timeval_subtract(&tvresult, &tvend, &tvbegin);
  procTime = tvresult.tv_usec + 1000000*tvresult.tv_sec;

  /* use MPI_Reduce to get time for process that took longest
   * when that's done, broadcast the total time to the root process
   */
  MPI_Reduce(&procTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&totalTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("(%i %i %i %i %i %i) total time: %lf usecs,\n",m,n,k,px,py,panel_size,totalTime);
  }
}

/** Program start */
int main(int argc, char *argv[]) {
  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME + 1];
  int namelen = 0;
  int ix,jx,kx; //looping vars
  struct timeval tv;

  MPI_Init(&argc, &argv); /* starts MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */
  MPI_Comm_size(MPI_COMM_WORLD, &np); /* Get number of processes */
  MPI_Get_processor_name(hostname, &namelen); /* Get hostname of node */
  printf("[Using Host:%s -- Rank %d out of %d]\n", hostname, rank, np);

  /* These tests use 64 processes */
  if (np != 64) {
    printf("Error: np=%d. Please use 64 processes\n", np);
  }

  /* Cases */
  int procs1[4] = {1,2,4,8};
  int procs2[4] = {64,32,16,8};
  int mdim[4] = {256,1024,256,1024};
  int ndim[4] = {256,256,256,1024};
  int kdim[4] = {256,256,1024,1024};
  int panelsizes[4] = {4,16,64,256};
    
  /* Run loop */
  for (ix=0; ix<4; ix++) { // loop over processor grides
      for (jx=0; jx<4; jx++) { // loop over dimensions
          for (kx=0; kx<4; kx++) { // loop over panelsizes

            //printf("%ix%i, %ix%ix%i, pb=%i\n",mdim[jx],
            //        ndim[jx],kdim[jx],procs1[ix],procs2[ix],panelsizes[kx]);
            random_matrix_test(mdim[jx],
                    ndim[jx],kdim[jx],procs1[ix],procs2[ix],panelsizes[kx]);
          }
      }
  }

  //exit_on_fail( random_matrix_test(1024, 1024, 1024, 8, 8, 1));

  finalize: MPI_Finalize();
  return 0;
}
