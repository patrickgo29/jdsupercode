#define NAIVE 0
#define OPENMP 1
#define MKL 2

void local_mm(const int m, const int n, const int k, const double alpha,
    const double *A, const int lda, const double *B, const int ldb,
    const double beta, double *C, const int ldc, int type);

