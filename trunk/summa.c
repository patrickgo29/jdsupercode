/**
 *  \file summa.c
 *  \brief Implementation of Scalable Universal 
 *    Matrix Multiplication Algorithm for Proj1
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "local_mm.h"

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
	int procGridX, int procGridY, int panel_size) {
//////////////////////////////////////////////////////////////////////////////
	//Setup MPI variables
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    //Setup time variables
    //One when broadcasting A, broadcasting B, and the local_mm
    double t1_bcast_A=0; double t2_bcast_A=0;
    double t1_bcast_B=0; double t2_bcast_B=0;
    double t1_local_mm=0; double t2_local_mm=0;
    double temp_bcast_A=0; double temp_bcast_B=0;
    double ttime_bcast_A=0; double ttime_bcast_B=0; double ttime_local_mm=0;
//////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
	//Calculate all the block dimensions
	//Ablock
	int height_Ablock = m / procGridX;
	int width_Ablock = k / procGridY;
	int size_Ablock = height_Ablock * width_Ablock;
	int y_Ablock_first = width_Ablock * y_proc;
	int y_Ablock_last = y_Ablock_first + width_Ablock - 1;
	//Bblock
	int height_Bblock = k / procGridX;
	int width_Bblock = n / procGridY;
	int size_Bblock = height_Bblock * width_Bblock;
	int x_Bblock_first = height_Bblock * x_proc;
	int x_Bblock_last = x_Bblock_first + height_Bblock - 1;
	//Cblock
	int height_Cblock = m / procGridX;
	int width_Cblock = n / procGridY;
	int size_Cblock = height_Cblock * width_Cblock;
///////////////////////////////////////////////////////////////////////////////
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
	//panel_buf_B_temp = (double*)malloc(size_pbB * sizeof(double));
	int x_pbB_first;
	int x_pbB_last;
///////////////////////////////////////////////////////////////////////////////
	if(!(panel_size > width_Ablock || panel_size > height_Bblock || (width_Ablock % panel_size != 0) || (height_Ablock % panel_size != 0))){
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

        //Bcast A. Compute time for processor 0 to complete and print
        if (rank == 0) {
            t1_bcast_A = MPI_Wtime();
        }

        MPI_Bcast(panel_buf_A, size_pbA, MPI_DOUBLE, first_row_rank, row_comm);

        if (rank == 0) {
            t2_bcast_A = MPI_Wtime();
            ttime_bcast_A = t2_bcast_A - t1_bcast_A;
            printf("(%i %i %i %i %i %i) time to broadcast A: %lf\n", m,n,k,
                    procGridX,procGridY,panel_size,ttime_bcast_A);
        }
		
        //Process all of the column data
                x_pbB_first = panel_size * i;
		x_pbB_last = x_pbB_first + panel_size - 1;
		int first_col_rank = x_pbB_first / height_Bblock;
		if(first_col_rank == col_rank) for(j = 0; j < width_Bblock; ++j){
                 	double* source_Bblock = Bblock + (x_pbB_first -  x_Bblock_first) + (j * height_Bblock);
                       	double* dest_pbB = panel_buf_B + (j * panel_size);
                       	memcpy(dest_pbB, source_Bblock, height_pbB * sizeof(double));
        }
		
        //Bcast B. Compute time for processor 0 to complete
        if (rank == 0) {
            t1_bcast_B = MPI_Wtime(); 
        }

        MPI_Bcast(panel_buf_B, size_pbB, MPI_DOUBLE, first_col_rank, col_comm);

        if (rank == 0) {
            t2_bcast_B = MPI_Wtime();
            ttime_bcast_B = t2_bcast_B - t1_bcast_B;
            printf("(%i %i %i %i %i %i) time to broadcast B: %lf\n", m,n,k,
                    procGridX,procGridY,panel_size,ttime_bcast_B);
        }
		
        //Local multiply.
        if (rank == 0) {
            t1_local_mm = MPI_Wtime();
        }

        local_mm(height_Cblock, width_Cblock, panel_size, 1.0,
                	panel_buf_A, height_pbA, panel_buf_B, height_pbB,
                	1.0, Cblock, height_Cblock);

        if (rank == 0) {
            t2_local_mm = MPI_Wtime();
            ttime_local_mm = t2_local_mm - t1_local_mm;
            printf("(%i %i %i %i %i %i) time to local_mm: %lf\n", m,n,k,
                    procGridX, procGridY,panel_size,ttime_local_mm);
        }
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}else{ 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
				if(j == last_row_rank)size_Bcast = panel_size * height_Ablock;
				else size_Bcast = (y_Jblock_last - y_pbA_first + 1) * height_Ablock;
			}else{
				dest_pbA = panel_buf_A + ((y_Jblock_first - y_pbA_first) * height_Ablock);
				if(j == last_row_rank) size_Bcast = (y_pbA_last - y_Jblock_first) * height_Ablock;
				else size_Bcast = size_Ablock;
			}
			if(j == row_rank){
				double* source_Ablock = Ablock + ((y_pbA_first - y_Jblock_first) * height_Ablock);
				memcpy(dest_pbA, source_Ablock, size_Bcast * sizeof(double));
			}

            // Broadcast A and compute total time to broadcast A
            if (rank == 0) {
                temp_bcast_A = 0;
                t1_bcast_A = MPI_Wtime();
            }
			
            MPI_Bcast(dest_pbA, size_Bcast, MPI_DOUBLE, j, row_comm); 

            if (rank == 0) {
                t2_bcast_A = MPI_Wtime();
                temp_bcast_A = t2_bcast_A - t1_bcast_A;
                ttime_bcast_A = ttime_bcast_A + temp_bcast_A;
            }

		}

        //Print total broadcast time for A
        if (rank == 0) {
             printf("(%i %i %i %i %i %i) time to broadcast A: %lf\n", m,n,k,
                    procGridX,procGridY,panel_size,ttime_bcast_A);
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
					if(j == col_rank) for(q = 0; q < width_pbB; q++)
						memcpy(dest_pbB + (q * size_Mcpy), source_loc + (q * height_Bblock),
							size_Mcpy * sizeof(double));
				}else{
					dest_pbB = panel_buf_B_temp;
					size_Mcpy = height_Bblock;
					size_Bcast = size_Mcpy * width_pbB;
					source_loc = Bblock;
					if(j == col_rank) for(q = 0; q< width_pbB; q++)
						memcpy(dest_pbB + (q * size_Mcpy), source_loc + (q * height_Bblock), 
							size_Mcpy * sizeof(double));
				}
			}

			//Broadcast the Bblock data to the column
            if (rank == 0) {
		      	temp_bcast_B = 0;
                t1_bcast_B = MPI_Wtime();
            }
            
            MPI_Bcast(dest_pbB, size_Bcast, MPI_DOUBLE, j, col_comm);

            if (rank == 0) {
                t2_bcast_B = MPI_Wtime();
                temp_bcast_B = t2_bcast_B - t1_bcast_B;
                ttime_bcast_B = ttime_bcast_B + temp_bcast_A;
            }
	
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

        //Print out time to broadcast B
        if (rank == 0) {
            printf("(%i %i %i %i %i %i) time to broadcast B: %lf\n", m,n,k,
                    procGridX,procGridY,panel_size,ttime_bcast_B);
        }


		//Perform the multiplication
        if (rank == 0) {
            t1_local_mm = MPI_Wtime();
        }
		
        local_mm(height_Cblock, width_Cblock, panel_size, 1.0,
                	panel_buf_A, height_pbA, panel_buf_B, height_pbB,
                	1.0, Cblock, height_Cblock);

        if (rank == 0) {
            t2_local_mm = MPI_Wtime();
            ttime_local_mm = t2_local_mm - t1_local_mm;
            printf("(%i %i %i %i %i %i) time to local_mm: %lf\n", m,n,k,
                    procGridX,procGridY,panel_size,ttime_local_mm);
        }
	}
	free(panel_buf_B_temp);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
///////////////////////////////////////////////////////////////////////////////
	//Make sure everyone has finished before finishing
	MPI_Barrier(MPI_COMM_WORLD);
///////////////////////////////////////////////////////////////////////////////
	//Clean up the memory to prevent leaks
	free(panel_buf_A);
	free(panel_buf_B);
}


