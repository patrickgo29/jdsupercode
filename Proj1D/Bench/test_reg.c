#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <nmmintrin.h> //SSE4
#include <math.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

static struct stopwatch_t *timer;
static struct stopwatch_t *timer2;

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
// note: A,B are 48x48 buffers, while C is the global pointer that is passed into matmult (for now 48)

void sse_kernel_eight_trans_dp(const int ldc, const int bs, const double *A, const double *B, double *C)
{
    int i;

    register __m128d A1 __asm__("xmm0"),
                     A2 __asm__("xmm1"),
                     A3 __asm__("xmm2"),
                     A4 __asm__("xmm3"),
                     B1 __asm__("xmm4"),
                     B2 __asm__("xmm5"),
                     B3 __asm__("xmm6"),
                     B4 __asm__("xmm7"),
                     C1 __asm__("xmm8"),
                     t1 __asm__("xmm9"),
                     t2 __asm__("xmm10"),
                     t3 __asm__("xmm11"),
                     t4 __asm__("xmm12");
    //load the first column of B
    B1 = _mm_load_pd(B);
    B2 = _mm_load_pd(B+2);
    B3 = _mm_load_pd(B+4);
    B4 = _mm_load_pd(B+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,49); //00110001
        t2 = _mm_dp_pd(A2,B2,49);
        t3 = _mm_dp_pd(A3,B3,49);
        t4 = _mm_dp_pd(A4,B4,49);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires the A registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the SECOND
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,50); //00110010
        t2 = _mm_dp_pd(A2,B2,50);
        t3 = _mm_dp_pd(A3,B3,50);
        t4 = _mm_dp_pd(A4,B4,50);
        //add results and update C1's second element
        //this is okay because we are guaranteed that the FIRST element
        //of t_i is 0
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
        _mm_store_pd(C+i,C1);
    }
    //load the second column of B
    B1 = _mm_load_pd(B+bs);
    B2 = _mm_load_pd(B+bs+2);
    B3 = _mm_load_pd(B+bs+4);
    B4 = _mm_load_pd(B+bs+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
         _mm_store_pd(C+i+ldc,C1);
    }
    //load the third column of B
    B1 = _mm_load_pd(B+bs*2);
    B2 = _mm_load_pd(B+bs*2+2);
    B3 = _mm_load_pd(B+bs*2+4);
    B4 = _mm_load_pd(B+bs*2+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*2);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
         _mm_store_pd(C+i+2*ldc,C1);
    }
    //load the fourth column of B
    B1 = _mm_load_pd(B+bs*3);
    B2 = _mm_load_pd(B+bs*3+2);
    B3 = _mm_load_pd(B+bs*3+4);
    B4 = _mm_load_pd(B+bs*3+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*3);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
         _mm_store_pd(C+i+3*ldc,C1);
    }
    //load the fifth column of B
    B1 = _mm_load_pd(B+bs*4);
    B2 = _mm_load_pd(B+bs*4+2);
    B3 = _mm_load_pd(B+bs*4+4);
    B4 = _mm_load_pd(B+bs*4+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*4);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
        _mm_store_pd(C+i+4*ldc,C1);
    }
    //load the sixth column of B
    B1 = _mm_load_pd(B+bs*5);
    B2 = _mm_load_pd(B+bs*5+2);
    B3 = _mm_load_pd(B+bs*5+4);
    B4 = _mm_load_pd(B+bs*5+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*5);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
         _mm_store_pd(C+i+5*ldc,C1);
    }
    //load the seventh column of B
    B1 = _mm_load_pd(B+bs*6);
    B2 = _mm_load_pd(B+bs*6+2);
    B3 = _mm_load_pd(B+bs*6+4);
    B4 = _mm_load_pd(B+bs*6+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*6);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
        _mm_store_pd(C+i+6*ldc,C1);
    }
    //load the eigth column of B
    B1 = _mm_load_pd(B+bs*7);
    B2 = _mm_load_pd(B+bs*7+2);
    B3 = _mm_load_pd(B+bs*7+4);
    B4 = _mm_load_pd(B+bs*7+6);
    for (i=0; i<8; i+=2) {
        //update first column by loading two columns of A per iteration
        C1 = _mm_load_pd(C+i+ldc*7);
        A1 = _mm_load_pd(A+i*bs); //first column
        A2 = _mm_load_pd(A+i*bs+2);
        A3 = _mm_load_pd(A+i*bs+4);
        A4 = _mm_load_pd(A+i*bs+6);
        //compute dot products
        t1 = _mm_dp_pd(A1,B1,0x31); //00110001
        t2 = _mm_dp_pd(A2,B2,0x31);
        t3 = _mm_dp_pd(A3,B3,0x31);
        t4 = _mm_dp_pd(A4,B4,0x31);
        //load second column of A, update second element
        //we save time if we prefetch now instead of waiting for 
        //C1 to get updated, since this only requires A_i registers
        A1 = _mm_load_pd(A+(i+1)*bs); //second column
        A2 = _mm_load_pd(A+(i+1)*bs+2);
        A3 = _mm_load_pd(A+(i+1)*bs+4);
        A4 = _mm_load_pd(A+(i+1)*bs+6);
        //add results and store in C1's first element
        //this is okay because we are guaranteed that the second
        //element of t_i is 0
        t1 = _mm_add_pd(t1,t2); //t1 and t3 get updated in parallel
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //now compute the dot products in parallel
        t1 = _mm_dp_pd(A1,B1,0x32); //00110010
        t2 = _mm_dp_pd(A2,B2,0x32);
        t3 = _mm_dp_pd(A3,B3,0x32);
        t4 = _mm_dp_pd(A4,B4,0x32);
        //add results and update C1's second element
        t1 = _mm_add_pd(t1,t2);
        t3 = _mm_add_pd(t3,t4);
        t2 = _mm_add_pd(t1,t3);
        C1 = _mm_add_pd(t2,C1);
        //store C1 back into the right place
        _mm_store_pd(C+i+7*ldc,C1);
    }
}

//4x4 SSE matrix multiply kernel. assumes A and B are contiguously loaded,
//while C is the global (non-contiguous) array
//uses hadd to accumulate add-mul results when done
//unrolls all loops fully
void sse_kernel_four_trans(const int bs, const int ldc, const double *A, const double *B, double *C)
{
    int i;
    //B1_copy and B2_copy are temp registers to expose parallelism
    //when updating the C array
    __m128d A1,A2,A3,A4,A5,A6,A7,A8,B1,B2,C1,C2,B1_copy,B2_copy;
    
    //fully load A
    A1 = _mm_load_pd(A);
    A2 = _mm_load_pd(A+2);
    A3 = _mm_load_pd(A+bs);
    A4 = _mm_load_pd(A+bs+2);
    A5 = _mm_load_pd(A+bs*2);
    A6 = _mm_load_pd(A+bs*2+2);
    A7 = _mm_load_pd(A+bs*3);
    A8 = _mm_load_pd(A+bs*3+2);

    //unroll the iterations
    //first column of C
    B1 = _mm_load_pd(B);
    B2 = _mm_load_pd(B+2);
    B1_copy = _mm_load_pd(B);
    B2_copy = _mm_load_pd(B+2);
    C1 = _mm_load_pd(C);
    C2 = _mm_load_pd(C+2);
    {
        __m128d temp1 = _mm_mul_pd(A1,B1);
        __m128d temp2 = _mm_mul_pd(A2,B2);
        __m128d temp3 = _mm_mul_pd(A3,B1_copy);
        __m128d temp4 = _mm_mul_pd(A4,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C1 = _mm_add_pd(C1,temp2);
    }
    {
        __m128d temp1 = _mm_mul_pd(A5,B1);
        __m128d temp2 = _mm_mul_pd(A6,B2);
        __m128d temp3 = _mm_mul_pd(A7,B1_copy);
        __m128d temp4 = _mm_mul_pd(A8,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C2 = _mm_add_pd(C2,temp2);
    }
//        C1 = _mm_add_pd(C1,
//         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
//                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
//    C2 = _mm_add_pd(C2,
//         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1_copy),_mm_mul_pd(A6,B2_copy)),
//                     _mm_add_pd(_mm_mul_pd(A7,B1_copy),_mm_mul_pd(A8,B2_copy))));
    _mm_store_pd(C,C1);
    _mm_store_pd(C+2,C2);

    //second column of C
    B1 = _mm_load_pd(B+bs);
    B2 = _mm_load_pd(B+bs+2);
    B1_copy = _mm_load_pd(B+bs);
    B2_copy = _mm_load_pd(B+bs+2);
    C1 = _mm_load_pd(C+ldc);
    C2 = _mm_load_pd(C+ldc+2);
    {
        __m128d temp1 = _mm_mul_pd(A1,B1);
        __m128d temp2 = _mm_mul_pd(A2,B2);
        __m128d temp3 = _mm_mul_pd(A3,B1_copy);
        __m128d temp4 = _mm_mul_pd(A4,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C1 = _mm_add_pd(C1,temp2);
    }
    {
        __m128d temp1 = _mm_mul_pd(A5,B1);
        __m128d temp2 = _mm_mul_pd(A6,B2);
        __m128d temp3 = _mm_mul_pd(A7,B1_copy);
        __m128d temp4 = _mm_mul_pd(A8,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C2 = _mm_add_pd(C2,temp2);
    }
    /*
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1_copy),_mm_mul_pd(A6,B2_copy)),
                     _mm_add_pd(_mm_mul_pd(A7,B1_copy),_mm_mul_pd(A8,B2_copy))));
    */
    _mm_store_pd(C+ldc,C1);
    _mm_store_pd(C+ldc+2,C2);
    
    //third column of C
    B1 = _mm_load_pd(B+2*bs);
    B2 = _mm_load_pd(B+2*bs+2);
    B1_copy = _mm_load_pd(B+2*bs);
    B2_copy = _mm_load_pd(B+2*bs+2);
    C1 = _mm_load_pd(C+2*ldc);
    C2 = _mm_load_pd(C+2*ldc+2);
    {
        __m128d temp1 = _mm_mul_pd(A1,B1);
        __m128d temp2 = _mm_mul_pd(A2,B2);
        __m128d temp3 = _mm_mul_pd(A3,B1_copy);
        __m128d temp4 = _mm_mul_pd(A4,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C1 = _mm_add_pd(C1,temp2);
    }
    {
        __m128d temp1 = _mm_mul_pd(A5,B1);
        __m128d temp2 = _mm_mul_pd(A6,B2);
        __m128d temp3 = _mm_mul_pd(A7,B1_copy);
        __m128d temp4 = _mm_mul_pd(A8,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C2 = _mm_add_pd(C2,temp2);
    }
    /*
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1_copy),_mm_mul_pd(A6,B2_copy)),
                     _mm_add_pd(_mm_mul_pd(A7,B1_copy),_mm_mul_pd(A8,B2_copy))));
    */
    _mm_store_pd(C+2*ldc,C1);
    _mm_store_pd(C+2*ldc+2,C2);

    //fourth column of C
    B1 = _mm_load_pd(B+3*bs);
    B2 = _mm_load_pd(B+3*bs+2);
    B1_copy = _mm_load_pd(B+3*bs);
    B2_copy = _mm_load_pd(B+3*bs+2);
    C1 = _mm_load_pd(C+3*ldc);
    C2 = _mm_load_pd(C+3*ldc+2);
    {
        __m128d temp1 = _mm_mul_pd(A1,B1);
        __m128d temp2 = _mm_mul_pd(A2,B2);
        __m128d temp3 = _mm_mul_pd(A3,B1_copy);
        __m128d temp4 = _mm_mul_pd(A4,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C1 = _mm_add_pd(C1,temp2);
    }
    {
        __m128d temp1 = _mm_mul_pd(A5,B1);
        __m128d temp2 = _mm_mul_pd(A6,B2);
        __m128d temp3 = _mm_mul_pd(A7,B1_copy);
        __m128d temp4 = _mm_mul_pd(A8,B2_copy);
        temp1 = _mm_add_pd(temp1,temp2);
        temp3 = _mm_add_pd(temp3,temp4);
        temp2 = _mm_hadd_pd(temp1,temp3);
        C2 = _mm_add_pd(C2,temp2);
    }
    /*
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1_copy),_mm_mul_pd(A6,B2_copy)),
                     _mm_add_pd(_mm_mul_pd(A7,B1_copy),_mm_mul_pd(A8,B2_copy))));
    */
    _mm_store_pd(C+3*ldc,C1);
    _mm_store_pd(C+3*ldc+2,C2);

/*
    for (i=0; i<4; i++) {
        //Assumes A is transposed
        //Load B1, B2
        B1 = _mm_load_pd(B+bs*i);
        B2 = _mm_load_pd(B+bs*i+2);
        C1 = _mm_load_pd(C+ldc*i);
        C2 = _mm_load_pd(C+ldc*i+2);
       
        //Compute C1, C2
        C1 = _mm_add_pd(C1,
             _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                         _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
        C2 = _mm_add_pd(C2,
             _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1),_mm_mul_pd(A6,B2)),
                         _mm_add_pd(_mm_mul_pd(A7,B1),_mm_mul_pd(A8,B2))));

        //Store back
        _mm_store_pd(&C[ldc*i],C1);
        _mm_store_pd(&C[ldc*i+2],C2);
    }
*/
}

void sse_kernel_four_trans_dp(const int bs, const int ldc, double *A, double *B, double *C)
{
    int i;
    __m128d A1,A2,A3,A4,A5,A6,A7,A8,B1,B2,C1,t1,t2,t3,t4;  
    //fully load A
    A1 = _mm_load_pd(A);
    A2 = _mm_load_pd(A+2);
    A3 = _mm_load_pd(A+bs);
    A4 = _mm_load_pd(A+bs+2);
    A5 = _mm_load_pd(A+2*bs);
    A6 = _mm_load_pd(A+2*bs+2);
    A7 = _mm_load_pd(A+3*bs);
    A8 = _mm_load_pd(A+3*bs+2);
/*
    for (i=0; i<4; i++) {
        B1 = _mm_load_pd(B+i*bs);
        B2 = _mm_load_pd(B+i*bs+2);
        C1 = _mm_load_pd(C+ldc*i);
        t1 = _mm_dp_pd(A1,B1,0x31);
        t2 = _mm_pd_pd(A2,B2,0x31);
        C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1); //update 1st
        t3 = _mm_dp_pd(A3,B1,0x32);
        t4 = _mm_dp_pd(A4,B2,0x32);
        C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); //update 2nd
        _mm_store_pd(C+ldc*i+j,C1);
        C1 = _mm_load_pd(C+ldc*i+2);
        t1 = _mm_dp_pd(A5,B1,0
    }
*/
    //unroll the iterations
    //first and second element of C
    B1 = _mm_load_pd(B);
    B2 = _mm_load_pd(B+2);
    C1 = _mm_load_pd(C);
    t1 = _mm_dp_pd(A1,B1,0x31); //store dot product in first element of t1
    t2 = _mm_dp_pd(A2,B2,0x31); //store dot product in first element of t2
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1); //update 1st
    t3 = _mm_dp_pd(A3,B1,0x32); //store dot product in sec element of t3
    t4 = _mm_dp_pd(A4,B2,0x32); //store dot product in sec element of t3
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); //update 2nd
    _mm_store_pd(C,C1);                    //store when done
    //third and fourth element of C
    C1 = _mm_load_pd(C+2);
    t1 = _mm_dp_pd(A5,B1,0x31); 
    t2 = _mm_dp_pd(A6,B2,0x31); 
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1); 
    t3 = _mm_dp_pd(A7,B1,0x32); 
    t4 = _mm_dp_pd(A8,B2,0x32); 
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1);
    _mm_store_pd(C+2,C1);                 
    //fifth and sixth element of C
    B1 = _mm_load_pd(B+bs);
    B2 = _mm_load_pd(B+bs+2);
    C1 = _mm_load_pd(C+ldc);
    t1 = _mm_dp_pd(A1,B1,0x31);
    t2 = _mm_dp_pd(A2,B2,0x31);
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1); 
    t3 = _mm_dp_pd(A3,B1,0x32);
    t4 = _mm_dp_pd(A4,B2,0x32);
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); 
    _mm_store_pd(C+ldc,C1);              
    //seventh and eighth element of C
    C1 = _mm_load_pd(C+ldc+2);
    t1 = _mm_dp_pd(A5,B1,0x31); 
    t2 = _mm_dp_pd(A6,B2,0x31);
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1);
    t3 = _mm_dp_pd(A7,B1,0x32);
    t4 = _mm_dp_pd(A8,B2,0x32); 
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); 
    _mm_store_pd(C+ldc+2,C1);  
    //ninth and tenth element of C
    B1 = _mm_load_pd(B+2*bs);
    B2 = _mm_load_pd(B+2*bs+2);
    C1 = _mm_load_pd(C+2*ldc);
    t1 = _mm_dp_pd(A1,B1,0x31);
    t2 = _mm_dp_pd(A2,B2,0x31);
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1);
    t3 = _mm_dp_pd(A3,B1,0x32);
    t4 = _mm_dp_pd(A4,B2,0x32); 
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); 
    _mm_store_pd(C+2*ldc,C1);  
    //11th and 12th element of C
    C1 = _mm_load_pd(C+2*ldc+2);
    t1 = _mm_dp_pd(A5,B1,0x31);
    t2 = _mm_dp_pd(A6,B2,0x31); 
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1);
    t3 = _mm_dp_pd(A7,B1,0x32);
    t4 = _mm_dp_pd(A8,B2,0x32); 
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1);
    _mm_store_pd(C+2*ldc+2,C1); 
    //13th and 14th element of C
    B1 = _mm_load_pd(B+3*bs);
    B2 = _mm_load_pd(B+3*bs+2);
    C1 = _mm_load_pd(C+3*ldc);
    t1 = _mm_dp_pd(A1,B1,0x31);
    t2 = _mm_dp_pd(A2,B2,0x31); 
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1);
    t3 = _mm_dp_pd(A3,B1,0x32); 
    t4 = _mm_dp_pd(A4,B2,0x32);
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1); 
    _mm_store_pd(C+3*ldc,C1);   
    //15th and 16th element of C
    C1 = _mm_load_pd(C+3*ldc+2);
    t1 = _mm_dp_pd(A5,B1,0x31); 
    t2 = _mm_dp_pd(A6,B2,0x31);
    C1 = _mm_add_pd(_mm_add_pd(t1,t2),C1);
    t3 = _mm_dp_pd(A7,B1,0x32);
    t4 = _mm_dp_pd(A8,B2,0x32);
    C1 = _mm_add_pd(_mm_add_pd(t3,t4),C1);
    _mm_store_pd(C+3*ldc+2,C1);               
}

void sse_kernel_four_trans_naive(const int bs, const int ldc, const double *A, const double *B, double *C)
{
  int i,j;
  __m128d A1,A2,B1,B2,C1,C2;
  for (i=0; i<4; i++)
  {
     A1 = _mm_load_pd(A);
     A2 = _mm_load_pd(A+2);
     B1 = _mm_set1_pd(B[i*bs]);
     C1 = _mm_mul_pd(A1,B1);
     C2 = _mm_mul_pd(A2,B1);
     for (j=1; j<4; j++) {
        A1 = _mm_load_pd(A+j*bs);
        A2 = _mm_load_pd(A+j*bs+2);
        B1 = _mm_set1_pd(B[i*bs+j]);
        C1 = _mm_add_pd(_mm_mul_pd(A1,B1),C1);
        C2 = _mm_add_pd(_mm_mul_pd(A2,B1),C2);
     }
     _mm_store_pd(&C[ldc*i],C1);
     _mm_store_pd(&C[ldc*i+2],C2);
  }
}
