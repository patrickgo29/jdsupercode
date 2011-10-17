#ifndef __MEM_FUNCTIONS_H__
#define __MEM_FUNCTIONS_H__

#include <xmmintrin.h>

//Assuming cache line size is 64Bytes
#define CACHE_LINE 64

#define L1 _MM_HINT_T0
#define L2 _MM_HINT_T1
#define L3 _MM_HINT_T2

void prefetch_Ablock(const double* A, int col, int row, int m, int n, int k, int bm, int bn, int bk);
void prefetch_Bblock(const double* B, int col, int row, int m, int n, int k, int bm, int bn, int bk);
void prefetch_Cblock(const double* C, int col, int row, int m, int n, int k, int bm, int bn, int bk);
void load_Abuffer(const double* A, double* Abuffer, int col, int row, int m, int n, int k, int bm, int bn, int bk);
void load_Bbuffer(const double* B, double* Bbuffer, int col, int row, int m, int n, int k, int bm, int bn, int bk);

#endif
