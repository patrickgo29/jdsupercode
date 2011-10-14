# David S. Noble, Jr.
# Matrix Multiply

all:
	@echo "=========================================================================="
	@echo "This Project's Makefile requires a specific build target:"
	@echo "    unittest_mm         : Build matrix multiply unittests"
	@echo "    unittest_summa      : Build summa unittests"
	@echo "    time_mm             : Build program to time local_mm"
	@echo "    time_summa          : Build program to time summa"
	@echo "    clean               : Removes generated files, junk"
	@echo "    clean-pbs           : Removes genereated pbs files"
	@echo "    run--unittest_mm    : Submit Test_Jobs/unittest_mm.pbs to the queue"
	@echo "    run--unittest_summa : Submit Test_Jobs/unittest_summa.pbs to the queue"
	@echo "    run--time_mm        : Submit Test_Jobs/time_mm.pbs to the queue"
	@echo "    run--time_summa     : Submit Test_Jobs/time_summa.pbs to the queue"
	@echo "    run--single_node    : Submit Test_Jobs/single_node.pbs to the queue"
	@echo "    run--multiple_nodes : Submit Test_Jobs/multiple_nodes.pbs to the queue"
	@echo "=========================================================================="

LINK_OPENMP_GCC = -fopenmp
LINK_MKL_GCC = -L/opt/intel/Compiler/11.1/059/mkl/lib/em64t/ \
               -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread

CC = mpicc
CFLAGS = -O -Wall -Wextra -lm $(LINK_MKL_GCC) $(LINK_OPENMP_GCC)

matrix_utils.o : matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -o $@ -c $<

local_mm.o : local_mm.c local_mm.h
	$(CC) $(CFLAGS) -o $@ -c $<

comm_args.o : comm_args.c comm_args.h
	$(CC) $(CFLAGS) -o $@ -c $<

summa.o : summa.c summa.h local_mm.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm.o : unittest_mm.c matrix_utils.h local_mm.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm : unittest_mm.o matrix_utils.o local_mm.o
	$(CC) $(CFLAGS) -o $@ $^

time_mm.o : time_mm.c matrix_utils.h local_mm.h comm_args.h
	$(CC) $(CFLAGS) -o $@ -c $<

time_mm : time_mm.o matrix_utils.o local_mm.o comm_args.o
	$(CC) $(CFLAGS) -o $@ $^

unittest_summa.o : unittest_summa.c matrix_utils.h local_mm.h summa.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_summa : unittest_summa.o matrix_utils.o local_mm.o summa.o
	$(CC) $(CFLAGS) -o $@ $^

time_summa.o : time_summa.c matrix_utils.h local_mm.h summa.h comm_args.h
	$(CC) $(CFLAGS) -o $@ -c $<

time_summa : time_summa.o matrix_utils.o local_mm.o summa.o comm_args.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY : clean
.PHONY : clean-pbs
	
clean : clean-pbs
	rm -f unittest_mm unittest_summa time_mm time_summa            
	rm -f *.o

clean-pbs : 
	@ [ -d Archive ] || mkdir -p ./Archive/ 
	@if [ -f unittest_mm.e* ]; then mv -f unittest_mm.e* ./Archive/; fi;
	@if [ -f unittest_mm.o* ]; then mv -f unittest_mm.o* ./Archive/; fi;
	@if [ -f unittest_summa.e* ]; then mv -f unittest_summa.e* ./Archive/; fi;
	@if [ -f unittest_summa.o* ]; then mv -f unittest_summa.o* ./Archive/; fi;
	@if [ -f time_mm.e* ]; then mv -f time_mm.e* ./Archive/; fi;
	@if [ -f time_mm.o* ]; then mv -f time_mm.o* ./Archive/; fi;
	@if [ -f time_summa.e* ]; then mv -f time_summa.e* ./Archive/; fi;
	@if [ -f time_summa.o* ]; then mv -f time_summa.o* ./Archive/; fi;
	@if [ -f single_node.e* ]; then mv -f single_node.e* ./Archive/; fi;
	@if [ -f single_node.o* ]; then mv -f single_node.o* ./Archive/; fi;
	@if [ -f multiple_nodes.e* ]; then mv -f multiple_nodes.e* ./Archive/; fi;
	@if [ -f multiple_nodes.o* ]; then mv -f multiple_nodes.o* ./Archive/; fi;

run--unittest_mm : unittest_mm clean-pbs
	qsub ./Test_Jobs/unittest_mm.pbs

run--unittest_summa : unittest_summa clean-pbs
	qsub ./Test_Jobs/unittest_summa.pbs

run--time_mm : time_mm clean-pbs
	qsub ./Test_Jobs/time_mm.pbs

run--time_summa : time_summa clean-pbs
	qsub ./Test_Jobs/time_summa.pbs

run--single_node : time_mm time_summa clean-pbs
	qsub ./Test_Jobs/single_node.pbs

run--multiple_nodes : time_summa clean-pbs
	qsub ./Test_Jobs/multiple_nodes.pbs

# eof
