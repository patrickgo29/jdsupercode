#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
   
#define REGISTER_SIZE 4

// The 4x4 and 8x8 SSE kernel. Takes a pointer to the start of the 4x4 matrices in 
// A and B that we want to multiply, and the start of the 4x4 matrix in C we want to store it in. 
// For example:
//
//
// A = |..........|
//     |..........|
//     |..........|
//     |..........|
//     |....Axxx..|
//     |....xxxx..|
//     |....xxxx..|
//     |....xxxx..|
//     |..........|
//     |..........|
//     |..........|
//
// the length of A (the larger matrix which contains the 4x4 submatrix we want) is lda
//
// note: A,B are 32x32 buffers, while C is the global pointer that is passed into matmult

void sse_kernel_four(const int ldc, const double *A, const double *B, double *C)
{

  int i,j;

  register __m128d A1 __asm__("xmm2"),
                   A2 __asm__("xmm3"),
                   B1 __asm__("xmm4"),
                   C1 __asm__("xmm5"),
                   C2 __asm__("xmm6");
      for (i=0; i<4; i++)
      {
          A1 = _mm_load_pd(A);
          A2 = _mm_load_pd(A+2);
          B1 = _mm_set1_pd(B[i*32]);

          C1 = _mm_mul_pd(A1,B1);
          C2 = _mm_mul_pd(A2,B1);
          for (j=1; j<4; j++) {
              A1 = _mm_load_pd(A+j*32);
              A2 = _mm_load_pd(A+j*32+2);
              B1 = _mm_set1_pd(B[i*32+j]);
              C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
              C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
          }
          _mm_store_pd(&C[ldc*i],C1);
          _mm_store_pd(&C[ldc*i+2],C2);
      }
}

void sse_kernel_eight(const int ldc, const double *A, const double *B, double *C)
{
    int i,j;

    register __m128d A1 __asm__("xmm2"),
                     A2 __asm__("xmm3"),
                     A3 __asm__("xmm4"),
                     A4 __asm__("xmm5"),
                     B1 __asm__("xmm6"),
                     C1 __asm__("xmm7"),
                     C2 __asm__("xmm8"),
                     C3 __asm__("xmm9"),
                     C4 __asm__("xmm10");

    for (i=0; i<8; i++) {
        //load As
        A1 = _mm_load_pd(A);
        A2 = _mm_load_pd(A+2);
        A3 = _mm_load_pd(A+4);
        A4 = _mm_load_pd(A+6);

        //load Bs
        B1 = _mm_set1_pd(B[i*32]);
    
        //perform calcs for first iteration
        C1 = _mm_mul_pd(A1,B1);
        C2 = _mm_mul_pd(A2,B1);
        C3 = _mm_mul_pd(A3,B1);
        C4 = _mm_mul_pd(A4,B1);
    
        //perform rest of iterations
        for (j=1; j<8; j++) {
            A1 = _mm_load_pd(A+j*32);
            A2 = _mm_load_pd(A+j*32+2);
            A3 = _mm_load_pd(A+j*32+4);
            A4 = _mm_load_pd(A+j*32+6);
            B1 = _mm_set1_pd(B[i*32+j]);
            C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
            C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
            C3 = _mm_add_pd(_mm_mul_pd(A3,B1),C3);
            C4 = _mm_add_pd(_mm_mul_pd(A4,B1),C4);
        } 
        
        // store accumulated dot products
        _mm_store_pd(&C[ldc*i],C1); 
        _mm_store_pd(&C[ldc*i+2],C2);
        _mm_store_pd(&C[ldc*i+4],C3);
        _mm_store_pd(&C[ldc*i+6],C4);
    }
}

void sse_kernel_back(const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
    int i,j,k;
    __m128d A1,A2,B1,C1,C2;
    int rs = REGISTER_SIZE;
    for (i=0; i<M; i+=rs) {
        for (j=0; j<N; j++) {
            A1 = _mm_load_pd(A+i);
            A2 = _mm_load_pd(A+i+2);
            B1 = _mm_set1_pd(B[j]);
               C1 = _mm_mul_pd(A1,B1);
               C2 = _mm_mul_pd(A2,B1);
               for (k=1; k<K; k++) {
                  A1 = _mm_load_pd(A+i+k*lda);
                  A2 = _mm_load_pd(A+i+k*lda+2);
                  B1 = _mm_set1_pd(B[j+k*lda]);
                  C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
                  C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
               }
               _mm_store_pd(C+i+j*lda,C1);
               _mm_store_pd(C+i+j*lda+2,C2);
         }
    }
}

int main()
{
    int i,j,k;
    int M,N,K;
    int lda;
    size_t copysize = sizeof(double)*32*32;
    size_t rbsize = sizeof(double)*16;
    double *A, *B, *C;
    
    // align memory
    posix_memalign((void**)&A,rbsize,copysize);
    posix_memalign((void**)&B,rbsize,copysize);
    posix_memalign((void**)&C,rbsize,copysize);

    // populate matrices with initial values
    for (i=0; i<64; i++)
    {
        A[i]=i;
        B[i]=i;
        C[i]=0;
    }
    
    // display results
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           A[i], A[i+8], A[i+16], A[i+24],
           A[i+32], A[i+40], A[i+48], A[i+56]);
    }
    printf("\n");
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           B[i], B[i+8], B[i+16], B[i+24],
           B[i+32], B[i+40], B[i+48], B[i+56]);
    }
    printf("\n");
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           C[i], C[i+8], C[i+16], C[i+24],
           C[i+32], C[i+40], C[i+48], C[i+56]);
    }
    printf("\n");

    // set constant block size of 4 for M,N,K
    // no need to consider fringe cases for now
//    M=4; N=4; K=4; lda=8;
//    for (i=0; i<lda; i+=4) {
//        for (j=0; j<lda; j+=4) {
//            for (k=0; k<lda; k+=4) {
//                printf("A Block (%i,%i) start: %lf\n",i,k,A[i+k*lda]);
//                printf("B Block (%i,%i) start: %lf\n",k,j,B[k+j*lda]);
//                printf("C Block (%i,%i) start: %lf\n",i,j,C[i+j*lda]);
//                test_sse(lda,M,N,K,A+i+k*lda,B+k+j*lda,C+i+j*lda);
//    }}}

    // run kernel
    sse_kernel(8,A,B,C);

    // display results
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           A[i], A[i+8], A[i+16], A[i+24],
           A[i+32], A[i+40], A[i+48], A[i+56]);
    }
    printf("\n");
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           B[i], B[i+8], B[i+16], B[i+24],
           B[i+32], B[i+40], B[i+48], B[i+56]);
    }
    printf("\n");
    for (i=0; i<8; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
           C[i], C[i+8], C[i+16], C[i+24],
           C[i+32], C[i+40], C[i+48], C[i+56]);
    }
    printf("\n");
    
}
