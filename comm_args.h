#ifndef __COMM_ARGS_H__
#define __COMM_ARGS_H__

typedef struct mat_mul_specs_struct{
	int m;
	int n;
	int k;
	int x;
	int y;
	int b;
	int type;
	int threads;
	int trials;
}mat_mul_specs;

mat_mul_specs * getMatMulSpecs(int argc, char **argv);

#endif
