/**
 *  \file summa.h
 *  \brief Implementation of Scalable Universal 
 *    Matrix Multiplication Algorithm for Proj1
 */

/**
 * Distributed Matrix Multiply using the SUMMA algorithm
 *  Computes C = A*B + C
 * 
 *  This function uses procGridX times procGridY processes
 *   to compute the product
 *  
 *  A is a m by k matrix, each process starts
 *	with a block of A (aBlock) 
 *  
 *  B is a k by n matrix, each process starts
 *	with a block of B (bBlock) 
 *  
 *  C is a m by n matrix, each process starts
 *	with a block of C (cBlock)
 *
 *  The resulting matrix is stored in C.
 *  A and B should not be modified during computation.
 * 
 *  Ablock, Bblock, and CBlock are stored in
 *   column-major format  
 *
 *  panel_size is the Panel Block Size
 **/

void summa(int m, int n, int k,
	   double *Ablock, double *Bblock, double *Cblock,
    	   int procGridX, int procGridY, int panel_size);

void summa_mms(mat_mul_specs * mms,
	       double *Ablock, double *Bblock, double *Cblock);

void summa_typed(int m, int n, int k, 
		 double *Ablock, double *Bblock, double *Cblock,
    		 int procGridX, int procGridY, int panel_size, 
		 int type);
