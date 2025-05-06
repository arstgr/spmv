# spmv
Subroutines optimized for sparse matrix/vector multiplication under high memory latency

## Compile the base spmv.c
```
gcc -O3 -fopenmp spmv.c -o spmv-base
```
## Compile optimized version
```
gcc -O3 -ftree-vectorize -funroll-loops -fprefetch-loop-arrays -falign-functions=64 -falign-loops=64 -funroll-all-loops -fopenmp -march=znver4 -fopt-info-vec-optimized -mavx512f -fprefetch-loop-arrays --param prefetch-latency=300 spmv_j.c -g -o spmv-znver4-opt
```

## To run
```
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close <executable> HV15R/HV15R.mtx
```
nproc could be set to 24, or a multiple of 8. A script, `run_test.sh' is also available to run under linux perf for profiling.

## Input datasets
The input datasets can be downloaded from 
```
https://sparse.tamu.edu/Fluorem/HV15R
```

and 

```
https://sparse.tamu.edu/Fluorem/PR02R
```
