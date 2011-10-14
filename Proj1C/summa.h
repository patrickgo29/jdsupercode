// David S. Noble, Jr.
// Matrix Multiply

#ifndef __SUMMA_H__
#define __SUMMA_H__

void summa(int m, int n, int k,
		   double *Ablock, double *Bblock, double *Cblock,
		   int procGridX, int procGridY, int panel_size);

void summa_mms(mat_mul_specs * mms,
			   double *Ablock, double *Bblock, double *Cblock);

void summa_typed(int m, int n, int k, 
				 double *Ablock, double *Bblock, double *Cblock,
				 int procGridX, int procGridY, int panel_size, 
				 int type);

#endif
