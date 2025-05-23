#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "mkl.h"
#include "mkl_spblas.h"

 
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

//base
void spmv_base(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}
// MKL
void spmv_mkl(sparse_matrix_t A_handle, const struct matrix_descr* descr, const double* x, double* y) {
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0, A_handle, *descr,
                    x, 0.0, y);
}

void l2_linf_error_check(const double* y, const double* z, int M) {
    double l2err = 0.0;
    double linf = 0.0;

    for (int i = 0; i < M; i++) {
        double diff = y[i] - z[i];
        l2err += diff * diff;
        linf += fabs(diff);
    }

    l2err = sqrt(l2err);

    printf("L2 norm Error between two methods  = %.6e\n", l2err);
    printf("Linf norm Error between two methods = %.6e\n", linf);
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
    // Confirm number of threads
    printf("MKL max threads: %d\n", mkl_get_max_threads());

    CSRMatrix* A = read_matrix_market(argv[1]);
    if (!A) return 1;

    //64 aligned
    int* row_ptr_1b = (int*)mkl_malloc((A->n_rows + 1) * sizeof(int), 64);
    int* col_idx_1b = (int*)mkl_malloc(A->n_nonzeros * sizeof(int), 64);
    for (int i = 0; i <= A->n_rows; ++i)
        row_ptr_1b[i] = A->row_ptr[i] + 1;
    for (int i = 0; i < A->n_nonzeros; ++i)
        col_idx_1b[i] = A->col_indices[i] + 1;

    // Create and optimize MKL sparse handle
    sparse_matrix_t A_handle;
    mkl_sparse_d_create_csr(&A_handle, SPARSE_INDEX_BASE_ONE,
                            A->n_rows, A->n_cols,
                            row_ptr_1b, row_ptr_1b + 1,
                            col_idx_1b, A->values);
    mkl_sparse_optimize(A_handle);
    struct matrix_descr descr = {.type = SPARSE_MATRIX_TYPE_GENERAL};


    double* x = (double*)mkl_malloc(A->n_cols * sizeof(double), 64);
    double* y = (double*)mkl_malloc(A->n_rows * sizeof(double), 64);
    double* y_base = (double*)mkl_malloc(A->n_rows * sizeof(double), 64);

    // Initialize input vector
    for (int i = 0; i < A->n_cols; i++) {
        x[i] = 1.0;
    }

    // Warm-up run
    spmv_mkl(A_handle, &descr, x, y);
    spmv_base(A, x, y_base);

    // Check L2 and Linf error
    l2_linf_error_check(y, y_base, A->n_rows);


    // Timing
    const int n_runs = 100;
    double start_time = get_time();
    
    for (int i = 0; i < n_runs; i++) {
        spmv_mkl(A_handle, &descr, x, y);
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
    mkl_free(x);
    mkl_free(y);
    mkl_free(y_base);
    mkl_free(row_ptr_1b);
    mkl_free(col_idx_1b);
    mkl_sparse_destroy(A_handle);

    return 0;
}