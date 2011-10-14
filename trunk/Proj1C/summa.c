// David S. Noble, Jr.
// Matrix Multiply

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

#include "comm_args.h"
#include "local_mm.h"
#include "summa.h"

void summa(int m, int n, int k,
	double *Ablock, double *Bblock, double *Cblock,
	int procGridX, int procGridY, int panel_size)
{
	summa_typed(m, n, k, Ablock, Bblock, Cblock,
				procGridX, procGridY, panel_size, NAIVE);
}

void summa_mms(mat_mul_specs * mms,
	double *Ablock, double *Bblock, double *Cblock)
{

	if(mms->type == OPENMP || mms->type == MKL)	
		omp_set_num_threads(mms->threads);
	
	summa_typed(mms->m, mms->n, mms->k, Ablock, Bblock, Cblock,
		mms->x, mms->y, mms->b, mms->type);

}

void summa_typed(int m, int n, int k,
	double *Ablock, double *Bblock, double *Cblock,
	int procGridX, int procGridY, int panel_size, int type)
{
					 
	//If not called within "summa_typed" then the number of threads
	//must be set by the OS environment variable OMP_NUM_THREADS

	//Double check to make sure this is a distributed summa calculation
	//If not, perform a single node local_mm and return
	if(procGridX == 1 && procGridY == 1 && panel_size == k){
		local_mm_typed(m, n, k, 1.0, Ablock, m, Bblock, k, 1.0, Cblock, m, type);
		return;
	}

	//Setup MPI variables
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//Determine x and y positions in 2-D array of processes
	int x_proc = rank % procGridX;
	int y_proc = (rank - x_proc) / procGridX;

	//Setup the column and row MPI groups for communication
	MPI_Group all_group, col_group, row_group;
	MPI_Comm_group(MPI_COMM_WORLD, &all_group);
	MPI_Comm row_comm, col_comm;
	int row_rank, col_rank, i, j;

	//Create row group and determine row_rank
	int row[procGridY];
	for(i = 0; i < procGridY; ++i) row[i] = x_proc + (i * procGridX);
	MPI_Group_incl(all_group, procGridY, row, &row_group);
	MPI_Comm_create(MPI_COMM_WORLD, row_group, &row_comm);
	MPI_Group_rank(row_group, &row_rank);

	//Create col group and determine col_rank
	int col[procGridX];
	for(i = 0; i < procGridX; ++i) col[i] = rank - x_proc + i;
	MPI_Group_incl(all_group, procGridX, col, &col_group);
	MPI_Comm_create(MPI_COMM_WORLD, col_group, &col_comm);
	MPI_Group_rank(col_group, &col_rank);

	//Calculate all the block dimensions
	//Ablock
	int height_Ablock = m / procGridX;
	int width_Ablock = k / procGridY;
	int size_Ablock = height_Ablock * width_Ablock;
	int y_Ablock_first = width_Ablock * y_proc;
	//Bblock
	int height_Bblock = k / procGridX;
	int width_Bblock = n / procGridY;
	int x_Bblock_first = height_Bblock * x_proc;
	//Cblock
	int height_Cblock = m / procGridX;
	int width_Cblock = n / procGridY;

	//Calculate and allocate for the panel buffers
	//Panel buffer A
	double* panel_buf_A;
	int height_pbA = height_Ablock;
	int width_pbA = panel_size;
	int size_pbA = height_pbA * width_pbA;
	panel_buf_A = (double*)malloc(size_pbA * sizeof(double));
	int y_pbA_first;
	int y_pbA_last;
	//Panel buffer B
	double* panel_buf_B;
	int height_pbB = panel_size;
	int width_pbB = width_Bblock;
	int size_pbB = height_pbB * width_pbB;
	panel_buf_B = (double*)malloc(size_pbB * sizeof(double));
	int x_pbB_first;
	int x_pbB_last;
	
	//Perform simple calculation if panel_size is evenly divisible into
	//the block sizes
	if(!(panel_size > width_Ablock || panel_size > height_Bblock || 
		(width_Ablock % panel_size != 0) || 
		(height_Ablock % panel_size != 0))){

		//Simple version, when panel_size divides into width_Ablock and height_Bblock
		for(i = 0; i < (k / panel_size); ++i){

			//Process all of the row data
			y_pbA_first = panel_size * i;
			y_pbA_last = y_pbA_first + panel_size - 1;
			int first_row_rank = y_pbA_first / width_Ablock;
			if(first_row_rank == row_rank){
				double* source_Ablock = Ablock + ((y_pbA_first - y_Ablock_first) * height_Ablock);
				memcpy(panel_buf_A, source_Ablock, size_pbA * sizeof(double));
			}
			MPI_Bcast(panel_buf_A, size_pbA, MPI_DOUBLE, first_row_rank, row_comm);

			//Process all of the column data
			x_pbB_first = panel_size * i;
			x_pbB_last = x_pbB_first + panel_size - 1;
			int first_col_rank = x_pbB_first / height_Bblock;
			if(first_col_rank == col_rank) for(j = 0; j < width_Bblock; ++j){
				double* source_Bblock = Bblock + (x_pbB_first -  x_Bblock_first) + (j * height_Bblock);
				double* dest_pbB = panel_buf_B + (j * panel_size);
				memcpy(dest_pbB, source_Bblock, height_pbB * sizeof(double));
			}

			MPI_Bcast(panel_buf_B, size_pbB, MPI_DOUBLE, first_col_rank, col_comm);

			local_mm_typed(height_Cblock, width_Cblock, panel_size, 1.0,
						   panel_buf_A, height_pbA, panel_buf_B, height_pbB,
						   1.0, Cblock, height_Cblock, type);

		}
	}else{
	//Must perform the more complicated calculation to take into account
	//local matrix multiplications that span multiple blocks
	
		double* panel_buf_B_temp = (double*)malloc(size_pbB * sizeof(double));
		for(i = 0; i < (k / panel_size); ++i){
			//Process all of the row data
			y_pbA_first = panel_size * i;
			y_pbA_last = y_pbA_first + panel_size - 1;
			int first_row_rank = y_pbA_first / width_Ablock;
			int last_row_rank = y_pbA_last / width_Ablock;
			for(j = first_row_rank; j <= last_row_rank; j++){
				
				int y_Jblock_first = width_Ablock * j;
				int y_Jblock_last = y_Jblock_first + width_Ablock - 1;
				double* dest_pbA;
				int size_Bcast;
				
				if(j == first_row_rank){
					dest_pbA = panel_buf_A;
				
					if(j == last_row_rank)
						size_Bcast = panel_size * height_Ablock;
					else 
						size_Bcast = (y_Jblock_last - y_pbA_first + 1) * height_Ablock;
				
				}else{
				
					dest_pbA = panel_buf_A + ((y_Jblock_first - y_pbA_first) * height_Ablock);
				
					if(j == last_row_rank)
						size_Bcast = (y_pbA_last - y_Jblock_first) * height_Ablock;
					else
						size_Bcast = size_Ablock;
				
				}
				
				if(j == row_rank){
					double* source_Ablock = Ablock + ((y_pbA_first - y_Jblock_first) * height_Ablock);
					memcpy(dest_pbA, source_Ablock, size_Bcast * sizeof(double));
				}
				
				MPI_Bcast(dest_pbA, size_Bcast, MPI_DOUBLE, j, row_comm); 
			}
			
			//Process all of the column data
			x_pbB_first = panel_size * i;
			x_pbB_last = x_pbB_first + panel_size - 1;
			int first_col_rank = x_pbB_first / height_Bblock;
			int last_col_rank = x_pbB_last / height_Bblock;
			for(j = first_col_rank; j <= last_col_rank; j++){
				
				int x_Jblock_first = height_Bblock * j;
				int x_Jblock_last = x_Jblock_first + height_Bblock - 1;
				double* dest_pbB;
				double* source_loc;
				double* store_dest;
				int size_Mcpy, size_Bcast, q;

				//Move the memory into buffers
				if(j == first_col_rank){
					if(j == last_col_rank){
						dest_pbB = panel_buf_B;
						size_Mcpy = height_pbB;
						size_Bcast = size_Mcpy * width_pbB;
						source_loc = Bblock + (x_pbB_first - x_Jblock_first);
						if(j == col_rank) for(q = 0; q < width_pbB; q++)
							memcpy(dest_pbB + (q * height_pbB), source_loc + (q * height_Bblock),
								   size_Mcpy * sizeof(double));
					}else{
						dest_pbB = panel_buf_B_temp;
						size_Mcpy = x_Jblock_last - x_pbB_first + 1;
						size_Bcast = size_Mcpy * width_pbB;
						source_loc = Bblock + (x_pbB_first - x_Jblock_first);
						if(j == col_rank) for(q = 0; q < width_pbB; q++)
							memcpy(dest_pbB + (q * size_Mcpy), source_loc + (q * height_Bblock),
								   size_Mcpy * sizeof(double));
					}
				}else{
					if(j == last_col_rank){
						dest_pbB = panel_buf_B_temp;
						size_Mcpy = x_pbB_last - x_Jblock_first + 1;
						size_Bcast = size_Mcpy * width_pbB;
						source_loc = Bblock;
						if(j == col_rank){
							for(q = 0; q < width_pbB; q++){
								memcpy(dest_pbB + (q * size_Mcpy), source_loc + (q * height_Bblock),
									   size_Mcpy * sizeof(double));
							}
						}
					}else{
						dest_pbB = panel_buf_B_temp;
						size_Mcpy = height_Bblock;
						size_Bcast = size_Mcpy * width_pbB;
						source_loc = Bblock;
						if(j == col_rank){
							for(q = 0; q< width_pbB; q++){
								memcpy(dest_pbB + (q * size_Mcpy), source_loc + (q * height_Bblock), 
									   size_Mcpy * sizeof(double));
							}
						}
					}
				}

				MPI_Bcast(dest_pbB, size_Bcast, MPI_DOUBLE, j, col_comm);
				
				//Store the memory from the temporary buffer
				if(j == first_col_rank){
					if(j != last_col_rank){
						store_dest = panel_buf_B;
						for(q = 0; q < width_pbB; q++)
						memcpy(store_dest + (q * height_pbB), dest_pbB + (q * size_Mcpy),
						size_Mcpy * sizeof(double));
					}
				}else{
					if(j == last_col_rank){
						store_dest = panel_buf_B + (x_Jblock_first - x_pbB_first);
						for(q = 0; q < width_pbB; q++)
							memcpy(store_dest + (q * height_pbB), dest_pbB + (q * size_Mcpy),
								   size_Mcpy * sizeof(double));
					}else{
						store_dest = panel_buf_B + (x_Jblock_first - x_pbB_first);
						for(q = 0; q < width_pbB; q++)
							memcpy(store_dest + (q * height_pbB), dest_pbB + (q * size_Mcpy),
								   size_Mcpy * sizeof(double));
					}
				}
			}

			//Perform the multiplication
			local_mm_typed(height_Cblock, width_Cblock, panel_size, 1.0,
						   panel_buf_A, height_pbA, panel_buf_B, height_pbB,
						   1.0, Cblock, height_Cblock, type);
		}
		
		//Clean up temp panel buff
		free(panel_buf_B_temp);
	}

	//Make sure everyone has finished before finishing
	MPI_Barrier(MPI_COMM_WORLD);

	//Clean up the memory to prevent leaks
	free(panel_buf_A);
	free(panel_buf_B);
}


