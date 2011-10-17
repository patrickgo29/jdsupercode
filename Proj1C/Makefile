# David S. Noble, Jr.
# Matrix Multiply

all:
	@echo "===================================================================="
	@echo "This Project's Makefile requires a specific build target:"
	@echo "    unittest_mm      : Build matrix multiply unittests"
	@echo "    time_mm          : Build program to time local_mm"
	@echo "    clean            : Removes generated files, junk"
	@echo "    clean-pbs        : Removes genereated pbs files"
	@echo "    run--unittest_mm : Submit Test_Jobs/unittest_mm.pbs to the queue"
	@echo "    run--time_mm     : Submit Test_Jobs/time_mm.pbs to the queue"
	@echo "    run--single_node : Submit Test_Jobs/single_node.pbs to the queue"
	@echo "===================================================================="

LINK_OPENMP_GCC = -fopenmp
LINK_MKL_GCC = -L/opt/intel/Compiler/11.1/059/mkl/lib/em64t/ \
               -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread

CC = mpicc
CFLAGS = -std=c99 -O -Wall -Wextra -lm $(LINK_MKL_GCC) $(LINK_OPENMP_GCC)

matrix_utils.o : matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -o $@ -c $<

local_mm.o : local_mm.c local_mm.h mem_functions.h
	$(CC) $(CFLAGS) -o $@ -c $<

mem_functions.o : mem_functions.c mem_functions.h
	$(CC) $(CFLAGS) -o $@ -c $<

comm_args.o : comm_args.c comm_args.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm.o : unittest_mm.c matrix_utils.h local_mm.h mem_functions.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm : unittest_mm.o matrix_utils.o local_mm.o mem_functions.o
	$(CC) $(CFLAGS) -o $@ $^

time_mm.o : time_mm.c matrix_utils.h local_mm.h comm_args.h mem_functions.h
	$(CC) $(CFLAGS) -o $@ -c $<

time_mm : time_mm.o matrix_utils.o local_mm.o comm_args.o mem_functions.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY : clean
.PHONY : clean-pbs
	
clean : clean-pbs
	rm -f unittest_mm time_mm            
	rm -f *.o

clean-pbs : 
	@ [ -d Archive ] || mkdir -p ./Archive/ 
	@if [ -f unittest_mm.e* ]; then mv -f unittest_mm.e* ./Archive/; fi;
	@if [ -f unittest_mm.o* ]; then mv -f unittest_mm.o* ./Archive/; fi;
	@if [ -f time_mm.e* ]; then mv -f time_mm.e* ./Archive/; fi;
	@if [ -f time_mm.o* ]; then mv -f time_mm.o* ./Archive/; fi;
	@if [ -f find_opt_blocking.e* ]; then mv -f find_opt_blocking.e* ./Archive/; fi;
	@if [ -f find_opt_blocking.o* ]; then mv -f find_opt_blocking.o* ./Archive/; fi;
	@if [ -f scaling_exps.e* ]; then mv -f scaling_exps.e* ./Archive/; fi;
	@if [ -f scaling_exps.o* ]; then mv -f scaling_exps.o* ./Archive/; fi;

run--unittest_mm : unittest_mm clean-pbs
	qsub6 ./Test_Jobs/unittest_mm.pbs

run--time_mm : time_mm clean-pbs
	qsub6 ./Test_Jobs/time_mm.pbs

run--find_opt_blocking : time_mm clean-pbs
	qsub6 ./Test_Jobs/find_opt_blocking.pbs

run--scaling_exps : time_mm clean-pbs
	qsub6 ./Test_Jobs/scaling_exps.pbs

# eof
