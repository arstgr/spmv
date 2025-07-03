#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <GraphBLAS.h>

#define MAX_LINE 1024

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    char line[MAX_LINE];
    // Skip headers and comments
    while (fgets(line, MAX_LINE, f)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions
    GrB_Index nrows, ncols, nnz;
    sscanf(line, "%lu %lu %lu", &nrows, &ncols, &nnz);

    // Allocate COO arrays
    GrB_Index *row_idx = malloc(nnz * sizeof(GrB_Index));
    GrB_Index *col_idx = malloc(nnz * sizeof(GrB_Index));
    double    *values  = malloc(nnz * sizeof(double));

    if (!row_idx || !col_idx || !values) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Read triplets
    for (GrB_Index k = 0; k < nnz; ++k) {
        int i, j;
        double x;
        if (fscanf(f, "%d %d %lf", &i, &j, &x) != 3) {
            fprintf(stderr, "Error reading entry %lu\n", k);
            return 1;
        }
        row_idx[k] = i - 1;  // Convert to 0-based
        col_idx[k] = j - 1;
        values[k] = x;
    }
    fclose(f);
    
    #pragma omp parallel
    {
        #pragma omp master
        {
            int threads = omp_get_num_threads();
            printf("Using %d OpenMP threads (from env or system default)\n", threads);
        }
    }	
	
    GrB_init(GrB_NONBLOCKING);

    GrB_Matrix A;
    GrB_Matrix_new(&A, GrB_FP64, nrows, ncols);
    GrB_Matrix_build(A, row_idx, col_idx, values, nnz, GrB_PLUS_FP64);

    GrB_Vector x, y;
    GrB_Vector_new(&x, GrB_FP64, ncols);
    GrB_Vector_new(&y, GrB_FP64, nrows);

    
    for (GrB_Index i = 0; i < ncols; i++) {
        GrB_Vector_setElement_FP64(x, 1.0, i);
    }


    GrB_mxv(y, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, x, NULL);
    GrB_wait(y, GrB_MATERIALIZE);  

    // Time the SpMV operation
    const int n_runs = 100;
    double start = omp_get_wtime();

    for (int i = 0; i < n_runs; i++) {
    	GrB_mxv(y, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, x, NULL);
    	GrB_wait(y, GrB_MATERIALIZE);  // Ensure computation completes
    }

    double end = omp_get_wtime();
    double elapsed_time = (end - start) / n_runs;
    printf("SPMV completed in %f seconds\n", elapsed_time);

    double gflops = (2.0 * nnz * 1e-9) /elapsed_time;
    printf("Performance: %f GFLOP/s\n", gflops);

    GrB_Vector_free(&x);
    GrB_Vector_free(&y);
    GrB_Matrix_free(&A);
    GrB_finalize();
    return 0;
}

