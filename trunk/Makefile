all:
	@echo "======================================================================"
	@echo "Proj 1: Distribued Matrix Multiply"
	@echo ""
	@echo "Valid build targets:"
	@echo ""
	@echo "           unittest_mm : Build matrix multiply unittests for Naive"
	@echo "        unittest_summa : Build summa unittests for Naive"
	@echo "               time_mm : Build program to time local_mm for Naive"
	@echo "            time_summa : Build program to time summa for Naive"
	@echo "      run--unittest_mm : Submit unittest_mm job"
	@echo "   run--unittest_summa : Submit unittest_summa job"
	@echo "          run--time_mm : Submit time_mm job"
	@echo "       run--time_summa : Submit time_summa job"
	@echo "                turnin : Create tarball with answers and results for T-Square"
	@echo "                 clean : Removes generated files, junk"
	@echo "             clean-pbs : Removes genereated pbs files"
	@echo "======================================================================"

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
	rm -f turnin.tar.gz

clean-pbs : 
	rm -f Proj1.*
#	@ [ -d archive ] || mkdir -p ./archive/
#	if [ -f Proj1.e* ]; then mv -f Proj1.e* ./archive/; fi;
#	if [ -f Proj1.o* ]; then mv -f Proj1.o* ./archive/; fi;

run--unittest_mm : unittest_mm clean-pbs
	qsub unittest_mm.pbs

run--unittest_summa : unittest_summa clean-pbs
	qsub unittest_summa.pbs

run--time_mm : time_mm clean-pbs
	qsub time_mm.pbs

run--time_summa : time_summa clean-pbs
	qsub time_summa.pbs

turnin : $(TURNIN_FILES)
	tar czvf turnin.tar.gz *

# eof
