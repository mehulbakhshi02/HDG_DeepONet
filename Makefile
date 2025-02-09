
# Petsc V<3.6
#include ${PETSC_DIR}/conf/variables
# Petsc V>=3.6
include ${PETSC_DIR}/lib/petsc/conf/variables


# COMPILER FLAGS
CFLAGS += -fpic #-fopenmp
CFLAGS += -g #-Wall
#CFLAGS += -DPARALLEL -std=c++11

#COPT += -O3 -march=native -msse -msse2 -msse3 -msse4.2
#COPT += -funroll-loops -fprefetch-loop-arrays 

# Standard HDG configuration
CFLAGS += -DHDG
#CFLAGS += -DLAXFRIEDRICH
CFLAGS += -DROE
#CFLAGS += -DLOCALDG
CFLAGS += -DBR2

CFLAGS += ${SFLAGS}

#CFLAGS += -O3
CFLAGS += ${COPT}
CFLAGS += ${PETSC_CC_INCLUDES}
#CFLAGS += -ftree-vectorizer-verbose=10
FFLAGS += ${PETSC_FC_INCLUDES}

# LINKER FLAGS FOR MAC
#LDFLAGS += -bundle -flat_namespace -undefined suppress #-lgfortran

# LINKER FLAGS FOR LINUX
LDFLAGS += -shared

# THE FOLLOWING DIRECTORIES NEED TO BE ADAPTED TO YOUR MACHINE
# NETGENDIR environment variable has to be setup correctly
TOPLEVEL = ${PWD}
FORTRAN = 'gfortran'
TEST_DIR = ./Test

src1 = LinearAlgebra/petsc_wrapper.cpp
src2 = main.cpp

obj1 = $(src1:%.cpp=%.o)
obj2 = $(src2:%.cpp=%.o)

.cpp.o:
	ngscxx -I/home/kunaghosh/anaconda3/envs/kunal/include/python3.10 -I/home/kunaghosh/anaconda3/envs/kunal/include/python3.10 -I. -c ${CFLAGS} -L$(TOPLEVEL) -L/home/kunaghosh/anaconda3/envs/kunal/lib/python3.10 -L/home/kunaghosh/anaconda3/envs/kunal/lib/python3.10/site-packages $< -o $@ -lpetsc_wrapper ${PETSC_KSP_LIB} -lpython3.10

default:
	echo $(TOPLEVEL)
	make all

all:
	make clean
	make petsc_wrapper.so
	make main.so
	cp main.so main.dylib
netgen:
	make all
	netgen -recent

clean:
	rm -f $(obj1) $(obj2) libpetsc_wrapper.a $(obj2:%.o=%.so) main.dylib

petsc_wrapper.so: $(obj1)
	ar r libpetsc_wrapper.a LinearAlgebra/petsc_wrapper.o
	ranlib libpetsc_wrapper.a

main.so: $(obj2)
	ngsld ${LDFLAGS} $(obj2) -L$(TOPLEVEL) -L/home/kunaghosh/anaconda3/envs/kunal/lib/python3.10 -L/home/kunaghosh/anaconda3/envs/kunal/lib/python3.10/site-packages -lsolve -lngfem -lngcomp -o $@ table_delaunay_wrapper.o table_tet_wrapper.o -lpetsc_wrapper ${PETSC_KSP_LIB} -lpython3.10

unit: 
	$(MAKE) clean -C $(TEST_DIR) -f Makefile.mac
	$(MAKE) -C $(TEST_DIR) -f Makefile.mac
	$(TEST_DIR)/all_unit

wrapper:
	${FORTRAN} -fpic -c table_delaunay_wrapper.f90
	${FORTRAN} -fpic -c table_tet_wrapper.f90
