/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclsparse.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Structure for CSR format
typedef struct {
    double *values;    // non-zero values
    aoclsparse_int *col_indices;  // column indices
    aoclsparse_int *row_ptr;      // row pointers
    int n_rows;        // number of rows
    int n_cols;        // number of columns
    int n_nonzeros;    // number of non-zero elements
} CSRMatrix;

// Function to read MatrixMarket format
CSRMatrix* read_matrix_market(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    // Skip comments
    char line[1024];
    do {
        fgets(line, 1024, fp);
    } while (line[0] == '%');

    // Read matrix dimensions and number of non-zeros
    CSRMatrix* mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    sscanf(line, "%d %d %d", &mat->n_rows, &mat->n_cols, &mat->n_nonzeros);

    // Allocate memory
    mat->values = (double *)malloc(mat->n_nonzeros * sizeof(double));
    mat->col_indices = (aoclsparse_int *)malloc(mat->n_nonzeros * sizeof(aoclsparse_int));
    mat->row_ptr = (aoclsparse_int *)calloc(mat->n_rows + 1, sizeof(aoclsparse_int));

    // Temporary arrays for COO format
    aoclsparse_int* row_indices = (aoclsparse_int *)malloc(mat->n_nonzeros * sizeof(aoclsparse_int));
    
    // Read entries
    for (int i = 0; i < mat->n_nonzeros; i++) {
        int row, col;
        double val;
        fscanf(fp, "%d %d %lf", &row, &col, &val);
        row_indices[i] = row- 1;  // Convert to 0-based indexing
        mat->col_indices[i] = col - 1;
        mat->values[i] = val;
    }

    // Convert COO to CSR
    for (int i = 0; i < mat->n_nonzeros; i++) {
        mat->row_ptr[row_indices[i] + 1]++;
    }
    
    for (int i = 0; i < mat->n_rows; i++) {
        mat->row_ptr[i + 1] += mat->row_ptr[i];
    }
    free(row_indices);
    fclose(fp);
    return mat;
}

void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}

// Get time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[])
{
    aoclsparse_status     status;
    aoclsparse_matrix     A     = NULL;
    aoclsparse_mat_descr  descr = NULL;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_index_base base  = aoclsparse_index_base_zero;

    double alpha = 1.0;
    double beta  = 0.0;

    if (argc != 2) {
        printf("Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    // Read matrix
    CSRMatrix* B = read_matrix_market(argv[1]);
    if (!B) return 1;

    int M = B->n_rows;
    int N = B->n_cols;
    int NNZ = B->n_nonzeros;
    
    aoclsparse_int *csr_row_ptr = B->row_ptr;
    aoclsparse_int *csr_col_ind = B->col_indices;
    double         *csr_val = B->values;

    // Input vectors
    double* x = (double*)malloc(B->n_cols * sizeof(double));
    double* y = (double*)malloc(B->n_rows * sizeof(double));
    double* z = (double*)malloc(B->n_rows * sizeof(double));

    // Initialize input vector
    for (int i = 0; i < B->n_cols; i++) {
        x[i] = 1.0;
    }

    printf("M=%d N=%d NNZ=%d\n", M,N, NNZ);
    printf("Starting the Base Case ..\n");
    // Warm-up run
    spmv(B, x, y);

    // Timing
    const int n_runs = 100;
    double start_time = get_time();
    
    for (int i = 0; i < n_runs; i++) {
        spmv(B, x, y);
    }
    
    double end_time = get_time();
    double elapsed_time = (end_time - start_time) / n_runs;

    // Calculate GFLOP/s
    // Each non-zero element requires 2 operations (multiply and add)
    double gflops = (2.0 * B->n_nonzeros * 1e-9) / elapsed_time;

    printf("Matrix dimensions: %d x %d\n", B->n_rows, B->n_cols);
    printf("Number of non-zeros: %d\n", B->n_nonzeros);
    printf("Average time per SpMV: %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);

    // Print aoclsparse version
    printf("%s\n", aoclsparse_get_version());

    // Create matrix descriptor
    // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(&descr);

    // Initialise sparse matrix
    status = aoclsparse_create_dcsr(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
    if(status != aoclsparse_status_success)
    {
        printf("Error while creating a sparse matrix, status = %i\n", status);
        return 1;
    }

    // Hint the system what operation to expect // to identify hint id(which routine is to be executed, destroyed later)
    status = aoclsparse_set_mv_hint(A, trans, descr, n_runs+1);
    if(status != aoclsparse_status_success)
    {
        printf("Error while hinting operation, status = %i\n", status);
        return 1;
    }

    // Optimize the matrix
    // currently AMD AOCL has a bug, where if you run the optimization call bellow, it affects the result of spmv op. 
    // commenting out the line below drops the performance but produces correct results
    status = aoclsparse_optimize(A);
    if(status != aoclsparse_status_success)
    {
        printf("Error while optimizing the matrix, status = %i\n", status);
        return 1;
    }

    // Invoke SPMV API (double precision)
    printf("Invoking aoclsparse_dmv...\n");
    status = aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, z);

    if(status != aoclsparse_status_success)
    {
        printf("Error while computing SPMV, status = %i\n", status);
        return 1;
    }

    // Timing
    start_time = get_time();
    for (int i = 0; i < n_runs; i++) {
        aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, z);
    }
    
    end_time = get_time();
    elapsed_time = (end_time - start_time) / n_runs;
    gflops = (2.0 * NNZ * 1e-9) / elapsed_time;

    printf("Matrix dimensions: %d x %d\n", M, N);
    printf("Number of non-zeros: %d\n", NNZ);
    printf("Average time per SpMV: %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);

    // calculating the norm2 error between 2 solutions
    double l2err=0.;
    for (int i=0;i<M;i++)
	    l2err += (y[i] - z[i])*(y[i] - z[i]);

    l2err = sqrt(l2err);
    printf("L2 norm Error between two methods = %f\n",l2err);

    double linf=0.;
    for (int i=0;i<M;i++)
            linf += fabs(y[i] - z[i]);

    printf("Linf norm Error between two methods = %f\n",linf);

    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    free(x);
    free(y);
    free(z);
    free(B->row_ptr);
    free(B->col_indices);
    free(B->values);
    return 0;
}
