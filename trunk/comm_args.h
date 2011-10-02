#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "local_mm.h"

typedef struct mat_mul_specs_struct{
	int m;
	int n;
	int k;
	int type;
}mat_mul_specs;

mat_mul_specs * getMatMulSpecs(int argc, char **argv);
