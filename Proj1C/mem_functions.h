#ifndef __MEM_FUNCTIONS_H__
#define __MEM_FUNCTIONS_H__

//Assuming cache line size is 64Bytes
#define CACHE_LINE 64

inline void prefetch_Ablock_K(double* A, int col, int m, int bk, int CACHE_LEVEL);
inline void prefetch_Ablock(double* A, int col, int row, int m, int bm, int bk, int CACHE_LEVEL);
inline void prefetch_Bblock(double* B, int col, int row, int k, int bk, int bn, int CACHE_LEVEL);
inline void prefetch_Cblock(double* C, int col, int row, int m, int bm, int bn, int CACHE_LEVEL);

#endif
