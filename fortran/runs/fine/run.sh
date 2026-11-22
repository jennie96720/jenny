#!/bin/bash
# converged DF run: unpolarised, then helicity (MPI, 4 processes)
export OPENBLAS_NUM_THREADS=1
MPI="mpirun --allow-run-as-root -np 4"
sed -i "s/^    mode        = '[a-z_]*',/    mode        = 'pdf',/" Configures.nlt
( time $MPI ../../debug/nucleon-qdq ) > pdf.log 2>&1
sed -i "s/^    mode        = '[a-z_]*',/    mode        = 'helicity',/" Configures.nlt
( time $MPI ../../debug/nucleon-qdq ) > helicity.log 2>&1
echo finished > done.flag
