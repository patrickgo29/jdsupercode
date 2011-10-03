#ifndef __LOCAL_MM_H__
#define __LOCAL_MM_H__

#include "comm_args.h"

#define NAIVE 0
#define OPENMP 1
#define MKL 2

void local_mm(const int m, const int n, const int k,
              const double alpha,
              const double *A, const int lda, 
              const double *B, const int ldb,
              const double beta, 
              double *C, const int ldc);

void local_mm_openmp(const int m, const int n, const int k,
              const double alpha,
              const double *A, const int lda, 
              const double *B, const int ldb,
              const double beta, 
              double *C, const int ldc);

void local_mm_mkl(const int m, const int n, const int k,
              const double alpha,
              const double *A, const int lda, 
              const double *B, const int ldb,
              const double beta, 
              double *C, const int ldc);

void local_mms(mat_mul_specs * mms,
                const double alpha,
                const double *A, const int lda,
                const double *B, const int ldb,
                const double beta,
                double *C, const int ldc);

void local_mm_typed(const int m, const int n, const int k,
                    const double alpha,
                    const double *A, const int lda,
                    const double *B, const int ldb,
                    const double beta,
                    double *C, const int ldc,
                    int type);

#endif
