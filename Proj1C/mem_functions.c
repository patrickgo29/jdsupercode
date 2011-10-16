#include <xmmintrin.h>

//Assuming cache line size is 64Bytes
#define CACHE_LINE 64

inline void prefetch_Ablock_K(double* A, int col, int m, int bk, int CACHE_LEVEL){
	double* A_prefetch = A + (col * m);
	for(int i = 0; i < (bk * m + CACHE_LINE - 1) / CACHE_LINE; i++){
		_mm_prefetch(A_prefetch, CACHE_LEVEL);
		A_prefetch += CACHE_LINE;
	}
}

inline void prefetch_Ablock(double* A, int col, int row, int m, int bm, int bk, int CACHE_LEVEL){
	double* A_prefetch = A + (col * m + row);
	for(int i = 0; i < bk; i++){
		double* A_prefetch_m = A_prefetch;
		for(int j = 0; j < (bm + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(A_prefetch_m, CACHE_LEVEL);
			A_prefetch_m += CACHE_LINE;
		}
		A_prefetch += m;
	}
}

inline void prefetch_Bblock(double* B, int col, int row, int k, int bk, int bn, int CACHE_LEVEL){
	double* B_prefetch = B + (col * k + row);
	for(int i = 0; i < bn; i++){
		double* B_prefetch_k = B_prefetch;
		for(int j = 0; j < (bk + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(B_prefetch_k, CACHE_LEVEL);
			B_prefetch_k += CACHE_LINE;
		}
		B_prefetch += k;
	}
}

inline void prefetch_Cblock(double* C, int col, int row, int m, int bm, int bn, int CACHE_LEVEL){
	double* C_prefetch = C + (col * m + row);
	for(int i = 0; i < bn; i++){
		double* C_prefetch_m = C_prefetch;
		for(int j = 0; j < (bm + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(C_prefetch_m, CACHE_LEVEL);
			C_prefetch_m += CACHE_LINE;
		}
		C_prefetch += m;
	}
}
