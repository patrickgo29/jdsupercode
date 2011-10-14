#include <stdio.h>
#include <stdlib.h>
#include <smmintrin.h>
#define REG_LINE 2
#define L1_LINE 64

// register kernel - compute A*B=C where A, B, C are 2x2
void mm_sse_reg(double *a, double *b, double *c)
{
    // initialize variables
    int i,j;
    __m128d a_temp, b_temp, c_temp;
   
    // loop over dot products for each element
    for (i=0; i<REG_LINE*REG_LINE; i+=REG_LINE) {
        c_temp = _mm_set1_pd(0.0);
        for (j=0; j<REG_LINE; j++)
        {
            a_temp = _mm_load_pd(&a[j*REG_LINE]);
            b_temp = _mm_set1_pd(b[i+j]);
            c_temp = _mm_add_pd(_mm_mul_pd(a_temp, b_temp), c_temp);
        }
        _mm_store_pd(&c[i], c_temp);
    }
}

// L1 cache blocking
// 2*(pb^2)*sizeof(double) = 32768 b (L1 cache size)
// => pb = 45.25 (closest power of 2 is 32)
//
// compute A*B where each line has 64 floats
void mm_sse_l1(double *a, double *b, double *r)
{
}

// L2 cache blocking
// 2*(pb^2)*sizeof(double) = 262144 b
// => pb = 128
void mm_sse_l2(double *a, double *b, double *r)
{
}

int main()
{

    double a[4] __attribute__((aligned(4))) = {4.0,4.0,
                                               4.0,4.0};
    double b[4] __attribute__((aligned(4))) = {4.0,4.0,
                                               4.0,4.0};
    double c[4] __attribute__((aligned(4)));

    mm_sse_reg(a,b,c);

    int i;
    for (i=0; i<4; i++)
    {
        printf("Value %d: %lf\n", i, c[i]);
    }

    return(0);
}
