// David S. Noble, Jr.
// Matrix Multiply

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "comm_args.h"

mat_mul_specs * getMatMulSpecs(int argc, char **argv){

	static struct option long_options[] = {
        	{"naive", 0, 0, 0},
        	{"openmp", 1, 0, 0},
        	{"mkl", 1, 0, 0},
		{"cbl", 0, 0, 0},
		{"cop", 0, 0, 0},
		{"trials", 1, 0, 0},
		{"bm", 1, 0, 0},
		{"bn", 1, 0, 0},
		{"bk", 1, 0, 0},
        	{NULL, 0, NULL, 0}
	};

	mat_mul_specs * mms;
	mms = (mat_mul_specs *)malloc(sizeof(mat_mul_specs));
	mms->type = 0; mms->threads = 1;
	mms->cbl = 0; mms->cop = 0;
	mms->trials = 0;
	mms->bm = 0; mms->bn = 0; mms->bk = 0;
	mms->m = 0; mms->n = 0; mms->k = 0;

	int c, option_index = 0, type_set = 0, trials_set = 0, abort_b= 0;
	while ((c = getopt_long(argc, argv, "m:n:k:", long_options, &option_index)) != -1) {
		switch (c) {
			case 0:
				switch (option_index){
					case 0:
						mms->type = option_index;
						mms->threads = 1;
						type_set = 1;
						break;
					case 1:
					case 2:
						mms->type = option_index;
						type_set = 1;
						if(optarg){
							mms->threads = atoi(optarg);
						}else{
							type_set = 1;
							mms->threads = 1;
							if(option_index == 1)
								printf("--openmp ");
							if(option_index == 2)
								printf("--mkl ");
							printf("was not passed a thread count, and is defaulting to 1!\n");
						}
						break;
					case 3:
						mms->cbl = 1;
						break;
					case 4:
						mms->cop = 1;
						break;
					case 5:
						mms->trials = atoi(optarg);
						trials_set = 1;
						break;
					case 6:
						mms->bm = atoi(optarg);
						break;
					case 7:
						mms->bn = atoi(optarg);
						break;
					case 8:
						mms->bk = atoi(optarg);
						break;
					default:
						//should never get here...
						break;
				}
				break;
			case 'm':
				mms->m = atoi(optarg);
				break;
			case 'n':
				mms->n = atoi(optarg);
				break;
			case 'k':
				mms->k = atoi(optarg);
				break;
			case ':':
				printf("argument %c requires a parameter\n", optopt);
				abort_b =1;
				break;
			default:
				break;
		}
	}

	if(!type_set){
		printf("multiplication type was not set: defaulting to naive implementation!\n");
	}

	if(!trials_set){
		printf("trials was not set: defaulting to 25 trials!\n");
	}

	if(mms->cbl){
		if(mms->type == 1){
			if(mms->bm == 0 || mms->bn == 0 || mms->bk == 0){
				printf("to use cache blocking(--cbl), arguments");
				if(mms->bm == 0)
					printf(" '--bm'");
				if(mms->bn == 0)
					printf(" '--bn'");
				if(mms->bk == 0)
					printf(" '--bk'");
				printf(" must be passed with a value greater than 0\n");
				abort_b= 1;
			}
		}else{
			printf("openmp(--openmp) must be enabled to use cache blocking(--cbl)\n");
			abort_b= 1;
		}
	}

	if(mms->cop){
		if(mms->type == 1){
			if(!mms->cbl){
				printf("cache blocking(--cbl) must be enabled to use the copy optimization(--cop)\n");
				abort_b= 1;
			}
		}else{
			printf("openmp(--openmp) ");
			if(!mms->cbl)
				printf("and cache blocking(--cbl) ");
			printf("must be enabled to use copy optimization(--cop)\n");
			abort_b= 1;
		}
	}

	if(mms->m == 0 || mms->n == 0 || mms->k == 0){
		printf("arguments");
		if(mms->m == 0)
			printf(" 'm' ");
		if(mms->n == 0)
			printf(" 'n' ");
		if(mms->k == 0)
			printf(" 'k' ");
		printf("must be passed with a value greater than 0\n");
		abort_b= 1;
	}
	
	if(abort_b){
		abort();
	}

	return mms;
}
