#include "comm_args.h"

mat_mul_specs * getMatMulSpecs(int argc, char **argv){
	static struct option long_options[] = {
        	{"naive", 0, 0, 0},
        	{"openmp", 0, 0, 0},
        	{"mkl", 0, 0, 0},
        	{NULL, 0, NULL, 0}
	};
	mat_mul_specs * mms;
	mms = (mat_mul_specs *)malloc(sizeof(mat_mul_specs));
	mms->m = 0; mms->n = 0; mms->k = 0; mms->type = 0; mms->t = 25;
	int c, option_index = 0, type_set = 0;
	while ((c = getopt_long(argc, argv, ":m:n:k:x:y:b:t:", long_options, &option_index)) != -1) {
		switch (c) {
			case 0:
				if(option_index >= 0 && option_index <= 2){
					mms->type = option_index;
					type_set = 1;
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
			case 'x':
				mms->x = atoi(optarg);
				break;
			case 'y':
				mms->y = atoi(optarg);
				break;
			case 'b':
				mms->b = atoi(optarg);
				break;
			case 't':
				mms->t = atoi(optarg);
				break;
			case ':':
				printf("argument %c requires a parameter\n", optopt);
				break;
			default:
				break;
		}
	}
	if(mms->m == 0 || mms->n == 0 || mms->k == 0){
		if(mms->m == 0){
			printf("argument 'm' must be passed with a value greater than 0\n");
		}
		if(mms->n == 0){
			printf("argument 'n' must be passed with a value greater than 0\n");
		}
		if(mms->k == 0){
			printf("argument 'k' must be passed with a value greater than 0\n");
		}
		abort();
	}
	if(!type_set){
		printf("multiplication type was not set!\n");
		printf("defaulting to naive implementation!\n");
	}
	return mms;
}
