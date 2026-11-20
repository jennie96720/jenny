# gfortran build.  'make FFLAGS="$(DEBUG)"' for a checked build while developing.
FC      = gfortran
FFLAGS ?= -O3 -march=native -fopenmp
DEBUG   = -O0 -g -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -Wall
LIBS    = -llapack -lblas
SRC     = src/kinds.f90 src/dirac.f90 src/quark.f90 src/quadrature.f90
OBJ     = $(patsubst src/%.f90,build/%.o,$(SRC))

all: build/test_step1

build/%.o: src/%.f90 | build
	$(FC) $(FFLAGS) -Jbuild -c $< -o $@

build/test_step1: tests/test_step1.f90 $(OBJ)
	$(FC) $(FFLAGS) -Jbuild $(OBJ) $< -o $@ $(LIBS)

build:
	mkdir -p build

clean:
	rm -rf build
