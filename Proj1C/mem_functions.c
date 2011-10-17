#include <string.h>

#include "mem_functions.h"

void prefetch_Ablock(const double* A, int col, int row, int m, int n, int k, int bm, int bn, int bk){
	double* A_prefetch = (double*)A + (col * m + row);
	for(int i = 0; i < bk; i++){
		double* A_prefetch_m = A_prefetch;
		for(int j = 0; j < (bm + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(A_prefetch_m, L2);
			A_prefetch_m += CACHE_LINE;
		}
		A_prefetch += m;
	}
}

void prefetch_Bblock(const double* B, int col, int row, int m, int n, int k, int bm, int bn, int bk){
	double* B_prefetch = (double*)B + (col * k + row);
	for(int i = 0; i < bn; i++){
		double* B_prefetch_k = B_prefetch;
		for(int j = 0; j < (bk + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(B_prefetch_k, L2);
			B_prefetch_k += CACHE_LINE;
		}
		B_prefetch += k;
	}
}

void prefetch_Cblock(const double* C, int col, int row, int m, int n, int k, int bm, int bn, int bk){
	double* C_prefetch = (double*)C + (col * m + row);
	for(int i = 0; i < bn; i++){
		double* C_prefetch_m = C_prefetch;
		for(int j = 0; j < (bm + CACHE_LINE - 1) / CACHE_LINE; j++){
			_mm_prefetch(C_prefetch_m, L2);
			C_prefetch_m += CACHE_LINE;
		}
		C_prefetch += m;
	}
}

void load_Abuffer(const double* A, double* Abuffer, int col, int row, int m, int n, int k, int bm, int bn, int bk){
	double* Acopy = (double*)A + (col * m + row);
	for(int i = 0; i < bk; i++){
		memcpy(Abuffer, Acopy, bm * sizeof(double));
		Abuffer += bm;
		Acopy += m;
	}
}

void load_Bbuffer(const double* B, double* Bbuffer, int col, int row, int m, int n, int k, int bm, int bn, int bk){
	double* Bcopy = (double*)B + row;
	for(int i = 0; i < n; ++i){
		memcpy(Bbuffer, Bcopy, bk * sizeof(double));
		Bbuffer += bk;
		Bcopy += m;
	}
}
