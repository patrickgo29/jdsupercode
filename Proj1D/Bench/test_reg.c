#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
   
#define REGISTER_SIZE 4

void test_sse (int lda, float *A, float *B, float *C)
{

  int i,j;
   __m128d A1,A2,B1,B2,C1,C2,C3,C4;
  for (i=0; i<16; i+=4)
  {
      A1 = _mm_load_pd(A);
      A2 = _mm_load_pd(A+2);

      b_line = _mm_set1_ps(b[i]);
      r_line = _mm_mul_ps(a_line,b_line);
      for (j=1; j<4; j++) {
          a_line = _mm_load_ps(&a[j*4]);
          b_line = _mm_set1_ps(b[i+j]);
          r_line = _mm_add_ps(_mm_mul_ps(a_line,b_line),r_line);
      }
      _mm_store_ps(&r[i],r_line);
  } 
}

void 
basic_dgemm (const int lda, const int M, const int N, const int K, double *A, double *B, double *C)
{
    int i,j,k;
    int ls = REGISTER_SIZE; //block size of 4
    
    // init __m128d datatypes 
    register __m128d A1 __asm__("xmm2"),
        A2 __asm__("xmm3"),
        B1 __asm__("xmm4"),
        B2 __asm__("xmm5"),
        B3 __asm__("xmm6"),
        B4 __asm__("xmm7"),
        C1 __asm__("xmm8"),
        C2 __asm__("xmm9"),
        C3 __asm__("xmm10"),
        C4 __asm__("xmm11"),
        C5 __asm__("xmm12"),
        C6 __asm__("xmm13"),
        C7 __asm__("xmm14"),
        C8 __asm__("xmm15");

    for(j=0; j<N; j+=ls) {
        for(i=0; i<M; i+=ls) {
           C += i+j*lda; // update C to point to correct block
           C1 = _mm_load_pd(C); C+=2;
           C2 = _mm_load_pd(C); C+=lda-2;
           C3 = _mm_load_pd(C); C+=2;
           C4 = _mm_load_pd(C); C+=lda-2;
           C5 = _mm_load_pd(C); C+=2;
           C6 = _mm_load_pd(C); C+=lda-2;
           C7 = _mm_load_pd(C); C+=2;
           C8 = _mm_load_pd(C);
           C += (-3*lda)-2; //reset C

           for(k=0; k<K; k+=1)
           {
                //load {A1,A2,B1,B2,B3,B4}
                A1 = _mm_load_pd(A+i+k*M);
                A2 = _mm_load_pd(A+i+k*M+2);
                B1 = _mm_load1_pd(B+j+k*N);
                B2 = _mm_load1_pd(B+j+k*N+1);
                B3 = _mm_load1_pd(B+j+k*N+2);
                B4 = _mm_load1_pd(B+j+k*N+3);

                //update Ci's
                C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
                C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
                C3 = _mm_add_pd(_mm_mul_pd(A1,B2),C3);
                C4 = _mm_add_pd(_mm_mul_pd(A2,B2),C4);
                C5 = _mm_add_pd(_mm_mul_pd(A1,B3),C5);
                C6 = _mm_add_pd(_mm_mul_pd(A2,B3),C6);
                C7 = _mm_add_pd(_mm_mul_pd(A1,B4),C7);
                C8 = _mm_add_pd(_mm_mul_pd(A2,B4),C8);
            }
            
            // move results to main memory
            _mm_store_sd(C,C1); C+=2;
            _mm_store_sd(C,C2); C+=lda-2;
            _mm_store_sd(C,C3); C+=2;
            _mm_store_sd(C,C4); C+=lda-2;
            _mm_store_sd(C,C5); C+=2;
            _mm_store_sd(C,C6); C+=lda-2;
            _mm_store_sd(C,C7); C+=2;
            _mm_store_sd(C,C8);
            C += (-3*lda)-2;

            // restore C for next iteration
            C -= i+j*lda;
        }
    }
}

int main()
{
    int i;
    size_t copysize = sizeof(double)*8*8;
    size_t rbsize = sizeof(float)*16;
//    double *A, *B, *C;
    float *A, *B, *C;
    
//    A = (double *)malloc(copysize);
//    B = (double *)malloc(copysize);
//    C = (double *)malloc(copysize);

    posix_memalign((void**)&A,rbsize,copysize);
    posix_memalign((void**)&B,rbsize,copysize);
    posix_memalign((void**)&C,rbsize,copysize);


    for (i=0; i<4*4; i++)
    {
        A[i]=i;
        B[i]=i;
        C[i]=0;
    }
    test_sse(A,B,C);
//    basic_dgemm(8,8,8,8,A,B,C);

    for (i=0; i<4; i++) {
        printf("%lf %lf %lf %lf\n",
                C[i], C[i+4], C[i+8], C[i+12]);
    }

//    for (i=0; i<8; i++) {
//        printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",
//                C[i],   C[i+8], C[i+16], C[i+24],
//                C[i+32],C[i+40],C[i+48], C[i+56]);
//    }

}
