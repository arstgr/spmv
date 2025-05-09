#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "mkl.h"  // Include MKL header

// Structure for CSR format
typedef struct {
    double *values;    // non-zero values
    int *col_indices;  // column indices
    int *row_ptr;      // row pointers
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
    mat->values = (double*)malloc(mat->n_nonzeros * sizeof(double));
    mat->col_indices = (int*)malloc(mat->n_nonzeros * sizeof(int));
    mat->row_ptr = (int*)calloc(mat->n_rows + 1, sizeof(int));

    // Temporary arrays for COO format
    int* row_indices = (int*)malloc(mat->n_nonzeros * sizeof(int));
    
    // Read entries
    for (int i = 0; i < mat->n_nonzeros; i++) {
        int row, col;
        double val;
        fscanf(fp, "%d %d %lf", &row, &col, &val);
        row_indices[i] = row - 1;  // Convert to 0-based indexing
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

// MKL-accelerated sparse matrix-vector multiplication
void spmv_mkl(const CSRMatrix* A, const double* x, double* y) {
    char trans = 'N';
    double alpha = 1.0;
    double beta = 0.0;

    mkl_dcsrmv(&trans, &(A->n_rows), &(A->n_cols), &alpha,
               "G**C", A->values, A->col_indices, A->row_ptr, A->row_ptr + 1,
               x, &beta, y);
}

// Get time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    // Read matrix
    CSRMatrix* A = read_matrix_market(argv[1]);
    if (!A) return 1;

    // Create input and output vectors
    double* x = (double*)malloc(A->n_cols * sizeof(double));
    double* y = (double*)malloc(A->n_rows * sizeof(double));

    // Initialize input vector
    for (int i = 0; i < A->n_cols; i++) {
        x[i] = 1.0;
    }

    // Warm-up run
    spmv_mkl(A, x, y);

    // Timing
    const int n_runs = 100;
    double start_time = get_time();
    
    for (int i = 0; i < n_runs; i++) {
        spmv_mkl(A, x, y);
    }
    
    double end_time = get_time();
    double elapsed_time = (end_time - start_time) / n_runs;

    // Calculate GFLOP/s
    double gflops = (2.0 * A->n_nonzeros * 1e-9) / elapsed_time;

    printf("Matrix dimensions: %d x %d\n", A->n_rows, A->n_cols);
    printf("Number of non-zeros: %d\n", A->n_nonzeros);
    printf("Average time per SpMV: %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);

    // Cleanup
    free(A->values);
    free(A->col_indices);
    free(A->row_ptr);
    free(A);
    free(x);
    free(y);

    return 0;
}