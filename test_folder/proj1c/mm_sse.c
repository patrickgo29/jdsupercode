#include <stdio.h>
#include <stdlib.h>
#include <smmintrin.h>
#define REG_LINE 4
#define L1_LINE 64

// register kernel - compute A*B=C where A, B, C are 4x4
void mm_sse_reg(float *a, float *b, float *c)
{
    // initialize variables
    int i,j;
    __m128 a_temp, b_temp, c_temp;
   
    // loop over dot products for each element
    for (i=0; i<REG_LINE*REG_LINE; i+=REG_LINE) {
        c_temp = _mm_set1_ps(0.0);
        for (j=0; j<REG_LINE; j++)
        {
            a_temp = _mm_load_ps(&a[j*REG_LINE]);
            b_temp = _mm_set1_ps(b[i+j]);
            c_temp = _mm_add_ps(_mm_mul_ps(a_temp, b_temp), c_temp);
        }
        _mm_store_ps(&c[i], c_temp);
    }
}

// L1 cache blocking
// 2*(pb^2)*sizeof(float) = 32768 b (L1 cache size)
// => pb = 64
//
// compute A*B where each line has 64 floats
void mm_sse_l1(float *a, float *b, float *r)
{
}

// L2 cache blocking
// 2*(pb^2)*sizeof(float) = 262144 b
// => pb = 181.01 (closest power of 2 is 128)
void mm_sse_l2(float *a, float*b, float *r)
{
}

int main()
{

    float a[16] __attribute__((aligned(16))) = {4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0};
    float b[16] __attribute__((aligned(16))) = {4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0,
                                                4.0,4.0,4.0,4.0};
    float c[16] __attribute__((aligned(16)));

    mm_sse_reg(a,b,c);

    int i;
    for (i=0; i<16; i++)
    {
        printf("Value %d: %lf\n", i, c[i]);
    }
    

    return(0);
}
