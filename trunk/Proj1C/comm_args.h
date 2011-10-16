// David S. Noble, Jr.
// Matrix Multiply

#ifndef __COMM_ARGS_H__
#define __COMM_ARGS_H__

typedef struct mat_mul_specs_struct{
	int type;
	int cbl;
	int cop;
	int threads;
	int trials;
	int l1;
	int l2;
	int l3;
	int m;
	int n;
	int k;
}mat_mul_specs;

mat_mul_specs * getMatMulSpecs(int argc, char **argv);

#endif
