#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <nmmintrin.h> //SSE4

#define BUFFER_SIZE 48
#define N_RUNS 2000

static struct stopwatch_t *timer;

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

void sse_kernel_eight_trans_dp(const int ldc, const double *A, const double *B, double *C)
{
    int i;
    int bs = BUFFER_SIZE;

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


void sse_kernel_four_trans_dp(const int ldc, const double *A, const double *B, double *C)
{
    int i;
    int bs = BUFFER_SIZE;

    register __m128d A1 __asm__("xmm0"),
                     A2 __asm__("xmm1"),
                     A3 __asm__("xmm2"),
                     A4 __asm__("xmm3"),
                     A5 __asm__("xmm4"),
                     A6 __asm__("xmm5"),
                     A7 __asm__("xmm6"),
                     A8 __asm__("xmm7"),
                     B1 __asm__("xmm8"),
                     B2 __asm__("xmm9"),
                     C1 __asm__("xmm10"),
                     t1 __asm__("xmm11"),
                     t2 __asm__("xmm12"),
                     t3 __asm__("xmm13"),
                     t4 __asm__("xmm14");
    
    //fully load A
    A1 = _mm_load_pd(A);
    A2 = _mm_load_pd(A+2);
    A3 = _mm_load_pd(A+bs);
    A4 = _mm_load_pd(A+bs+2);
    A5 = _mm_load_pd(A+2*bs);
    A6 = _mm_load_pd(A+2*bs+2);
    A7 = _mm_load_pd(A+3*bs);
    A8 = _mm_load_pd(A+3*bs+2);

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

void sse_kernel_four_trans(const int ldc, const double *A, const double *B, double *C)
{
    int i;
    int bs = BUFFER_SIZE;

    register __m128d A1 __asm__("xmm1"),
                     A2 __asm__("xmm2"),
                     A3 __asm__("xmm3"),
                     A4 __asm__("xmm4"),
                     A5 __asm__("xmm5"),
                     A6 __asm__("xmm6"),
                     A7 __asm__("xmm7"),
                     A8 __asm__("xmm8"),
                     B1 __asm__("xmm9"),
                     B2 __asm__("xmm10"),
                     C1 __asm__("xmm11"),
                     C2 __asm__("xmm12");
    
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
    //unroll the iterations
    //first column of C
    B1 = _mm_load_pd(B);
    B2 = _mm_load_pd(B+2);
    C1 = _mm_load_pd(C);
    C2 = _mm_load_pd(C+2);
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1),_mm_mul_pd(A6,B2)),
                     _mm_add_pd(_mm_mul_pd(A7,B1),_mm_mul_pd(A8,B2))));
    _mm_store_pd(C,C1);
    _mm_store_pd(C+2,C2);

    //second column of C
    B1 = _mm_load_pd(B+bs);
    B2 = _mm_load_pd(B+bs+2);
    C1 = _mm_load_pd(C+ldc);
    C2 = _mm_load_pd(C+ldc+2);
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1),_mm_mul_pd(A6,B2)),
                     _mm_add_pd(_mm_mul_pd(A7,B1),_mm_mul_pd(A8,B2))));
    _mm_store_pd(C+ldc,C1);
    _mm_store_pd(C+ldc+2,C2);
    
    //third column of C
    B1 = _mm_load_pd(B+2*bs);
    B2 = _mm_load_pd(B+2*bs+2);
    C1 = _mm_load_pd(C+2*ldc);
    C2 = _mm_load_pd(C+2*ldc+2);
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1),_mm_mul_pd(A6,B2)),
                     _mm_add_pd(_mm_mul_pd(A7,B1),_mm_mul_pd(A8,B2))));
    _mm_store_pd(C+2*ldc,C1);
    _mm_store_pd(C+2*ldc+2,C2);

    //fourth column of C
    B1 = _mm_load_pd(B+3*bs);
    B2 = _mm_load_pd(B+3*bs+2);
    C1 = _mm_load_pd(C+3*ldc);
    C2 = _mm_load_pd(C+3*ldc+2);
    C1 = _mm_add_pd(C1,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A1,B1),_mm_mul_pd(A2,B2)),
                     _mm_add_pd(_mm_mul_pd(A3,B1),_mm_mul_pd(A4,B2))));
    C2 = _mm_add_pd(C2,
         _mm_hadd_pd(_mm_add_pd(_mm_mul_pd(A5,B1),_mm_mul_pd(A6,B2)),
                     _mm_add_pd(_mm_mul_pd(A7,B1),_mm_mul_pd(A8,B2))));
    _mm_store_pd(C+3*ldc,C1);
    _mm_store_pd(C+3*ldc+2,C2);
*/
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

}

// uses 4x4 SSE kernel to compute the product of two matrices with dimensions [M,K] and [K,N] and stores it into a matrix [M,N]
// enforce constraint that (M,N,K) = (48,48,4) or (4,4,48)
// works with any buffer size as above
void block_mult_four(const int ldc, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
    //check if M==N
    if (M==N) {
        int i,j,k;
        int bs = BUFFER_SIZE;
        int dim = M; //M and N are the same
        //now check if M<K, if it is, this is a row fringe
        if (dim<K) {

        }
        //otherwise it's a column fringe
        if (dim>K) {
//            for (i=0; i<dim; i+=4) {     //for each block in A
//               for (j=0; i<dim; j+=4) { //for each block in B
//                    //need to compute (dim/4)*(dim/4) blocks in C
//                    sse_kernel_four_trans(ldc,
//                                          A+i,
//                                          B+
//        }}}
        }
        //catch the case where M==N==K
        if (M==N && M==K && N==K) {
            int dim = M; //same dimensions
            int i,j,k;
            int bs = BUFFER_SIZE;
            for (i=0; i<dim; i+=4) { // twelve blocks on x-axis
                for (j=0; j<dim; j+=4) { // twelve blocks on y-axis
                    // compute the (i,j)th block
                    // there are 144 blocks we need to compute
                    for (k=0; k<dim; k+=4) {
//                        sse_kernel_four_trans(ldc, 
//                                              A+j*bs+k, 
//                                              B+i*bs+k, 
//                                              C+j+i*ldc);
                          sse_kernel_four_trans_dp(ldc,
                                                   A+j*bs+k,
                                                   B+i*bs+k,
                                                   C+j+i*ldc);
    }}}}}
    else {
        printf("M!=N\n");
    }
}

void block_mult_eight(const int ldc, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
    //check if M==N
    if (M==N) {
        int i,j,k;
        int bs = BUFFER_SIZE;
        int dim = M; //M and N are the same
        //now check if M<K, if it is, this is a row fringe
        if (dim<K) {

        }
        //otherwise it's a column fringe
        if (dim>K) {
//            for (i=0; i<dim; i+=4) {     //for each block in A
//               for (j=0; i<dim; j+=4) { //for each block in B
//                    //need to compute (dim/4)*(dim/4) blocks in C
//                    sse_kernel_four_trans(ldc,
//                                          A+i,
//                                          B+
//        }}}
        }
        //catch the case where M==N==K
        if (M==N && M==K && N==K) {
            int dim = M; //same dimensions
            int i,j,k;
            int bs = BUFFER_SIZE;
            for (i=0; i<dim; i+=8) { // twelve blocks on x-axis
                for (j=0; j<dim; j+=8) { // twelve blocks on y-axis
                    // compute the (i,j)th block
                    // there are 144 blocks we need to compute
                    for (k=0; k<dim; k+=8) {
//                        sse_kernel_four_trans(ldc, 
//                                              A+j*bs+k, 
//                                              B+i*bs+k, 
//                                              C+j+i*ldc);
                          sse_kernel_four_trans_dp(ldc,
                                                   A+j*bs+k,
                                                   B+i*bs+k,
                                                   C+j+i*ldc);
    }}}}}
    else {
        printf("M!=N\n");
    }
}

//this works but assumes A is not transposed
//also does some scalar operations so presumably is not as efficient
void sse_kernel_four(const int ldc, const double *A, const double *B, double *C)
{

  int i,j;
  int bs = BUFFER_SIZE;

  register __m128d A1 __asm__("xmm2"),
                   A2 __asm__("xmm3"),
                   B1 __asm__("xmm4"),
                   C1 __asm__("xmm5"),
                   C2 __asm__("xmm6");
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

//this doesn't work
void sse_kernel_eight(const int ldc, const double *A, const double *B, double *C)
{
    int i,j;
    int bs = BUFFER_SIZE;

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
        B1 = _mm_set1_pd(B[i*bs]);
    
        //perform calcs for first iteration
        C1 = _mm_mul_pd(A1,B1);
        C2 = _mm_mul_pd(A2,B1);
        C3 = _mm_mul_pd(A3,B1);
        C4 = _mm_mul_pd(A4,B1);
    
        //perform rest of iterations
        for (j=1; j<8; j++) {
            A1 = _mm_load_pd(A+j*bs);
            A2 = _mm_load_pd(A+j*bs+2);
            A3 = _mm_load_pd(A+j*bs+4);
            A4 = _mm_load_pd(A+j*bs+6);
            B1 = _mm_set1_pd(B[i*bs+j]);
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


int main()
{
    int i,j,k;
    int M,N,K;
    int lda;
    clock_t start, end;
    size_t copysize = sizeof(double)*BUFFER_SIZE*BUFFER_SIZE;
    size_t rbsize = sizeof(double);
    double *A, *B, *C;
    
    // align memory
    posix_memalign((void**)&A,rbsize,copysize);
    posix_memalign((void**)&B,rbsize,copysize);
    posix_memalign((void**)&C,rbsize,copysize);

    // populate (square) matrices with initial values
    // give some room for padding
//    M=16; N=16; K=16;
    M=BUFFER_SIZE; N=BUFFER_SIZE; K=BUFFER_SIZE;
    for (i=0; i<BUFFER_SIZE*BUFFER_SIZE; i++)
    {
        A[i]=i;
        B[i]=i;
        C[i]=0;
    }
    
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

    // run kernel and time it
    start = clock();
    for (i=0; i<N_RUNS; i++) {
//      sse_kernel_four_trans(BUFFER_SIZE,A,B,C);
//      sse_kernel_four_trans_dp(BUFFER_SIZE,A,B,C);
        block_mult_four(BUFFER_SIZE,M,N,K,A,B,C);
//        sse_kernel_eight_trans_dp(BUFFER_SIZE,A,B,C);
    }
    end = clock();
    double dif = (1./CLOCKS_PER_SEC)*((double)start+(double)end);
    printf("Total time was %lf\n",dif);
    printf("Each routine took %lf\n", dif/(double)N_RUNS);

//    stopwatch_destroy(timer);

/*    // display results
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
*/

    // display first and last column
    printf("First column:\n");
    for (i=0; i<BUFFER_SIZE; i++) {
        printf("%lf\n",C[i]);
    }
    printf("Last column:\n");
    for (i=0; i<BUFFER_SIZE; i++) {
        printf("%lf\n",C[i+BUFFER_SIZE*(BUFFER_SIZE-1)]);
    } 

}

