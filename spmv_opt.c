#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdalign.h>
#include <immintrin.h>


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
    
    size_t alignment = 64;
    size_t nnz_size = mat->n_nonzeros * sizeof(double);
    if (nnz_size % alignment != 0) {
        nnz_size += alignment - (nnz_size % alignment);
    }
    size_t nnz_isize = mat->n_nonzeros * sizeof(int);
    if (nnz_isize % alignment != 0) {
	    nnz_isize += alignment - (nnz_isize % alignment);
    }
    size_t nr_size = (mat->n_rows + 1) * sizeof(int);
    if (nr_size % alignment != 0) {
	nr_size += alignment - (nr_size % alignment);
    }

    // Allocate memory
    mat->values = (double*)aligned_alloc(alignment, nnz_size); //(double*)malloc(mat->n_nonzeros * sizeof(double));
    mat->col_indices = (int*)aligned_alloc(alignment, nnz_isize);//(int*)malloc(mat->n_nonzeros * sizeof(int));
    mat->row_ptr = (int*)aligned_alloc(alignment, nr_size) ;//(int*)calloc(mat->n_rows + 1, sizeof(int));

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

// Sparse matrix-vector multiplication
/*void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}*/
/*void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
	    double sum = 0.0;
	    for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j += 4) {
    		double sum_part = 0.0;
    		sum_part += A->values[j] * x[A->col_indices[j]];
    		if (j + 1 < A->row_ptr[i + 1])
    		    sum_part += A->values[j + 1] * x[A->col_indices[j + 1]];
    		if (j + 2 < A->row_ptr[i + 1])
    		    sum_part += A->values[j + 2] * x[A->col_indices[j + 2]];
    		if (j + 3 < A->row_ptr[i + 1])
    		    sum_part += A->values[j + 3] * x[A->col_indices[j + 3]];
    		sum += sum_part;
		}
	    y[i] = sum;
    }
}*/
/*void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
	    if (j%8==0){
	    	__builtin_prefetch(&A->values[j+4] , 0, 1);
	    	__builtin_prefetch(&x[A->col_indices[j+1]] , 0, 1);
		__builtin_prefetch(&x[A->col_indices[j+2]] , 0, 1);
		__builtin_prefetch(&x[A->col_indices[j+3]] , 0, 1);
		__builtin_prefetch(&x[A->col_indices[j+4]] , 0, 1);
	        __builtin_prefetch(&A->col_indices[j+8] , 0, 1);
	    }
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}
*/
/*void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        __m256d vec_sum = _mm256_setzero_pd(); // Vectorized sum (4 doubles)

        int j = A->row_ptr[i];
        int end = A->row_ptr[i + 1];
        
        // Process in chunks of 4
        for (; j <= end - 4; j += 4) {
            __m256d vec_values = _mm256_load_pd(&A->values[j]); // Load 4 values
            __m256d vec_x = _mm256_set_pd(
                x[A->col_indices[j+3]],
                x[A->col_indices[j+2]],
                x[A->col_indices[j+1]],
                x[A->col_indices[j]]
            ); // Load x values
            vec_sum = _mm256_fmadd_pd(vec_values, vec_x, vec_sum); // Multiply and accumulate
        }

        // Handle the remaining elements
        double sum = 0.0;
        for (; j < end; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }

        // Store back the horizontal sum of vec_sum and add the remainder
        double sum_array[4];
        _mm256_storeu_pd(sum_array, vec_sum); // Store back to memory
        y[i] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum;
    }
}*/
/*
void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        __m256d vec_sum = _mm256_setzero_pd(); // Vectorized sum (4 doubles)

        int j = A->row_ptr[i];
        int end = A->row_ptr[i + 1];
        
        // Process in chunks of 4
        for (; j <= end - (end%4); j += 4) {
	    // Prefetch future values and indices
            __builtin_prefetch(&A->values[j + 8], 0, 1);
            __builtin_prefetch(&A->col_indices[j + 16], 0, 2);

	    __builtin_prefetch(&x[A->col_indices[j + 8]], 0, 1);
            
            __m256d vec_values = _mm256_load_pd(&A->values[j]); // Load 4 values
            __m256d vec_x = _mm256_set_pd(
                x[A->col_indices[j+3]],
                x[A->col_indices[j+2]],
                x[A->col_indices[j+1]],
                x[A->col_indices[j]]
            ); // Load x values
            vec_sum = _mm256_fmadd_pd(vec_values, vec_x, vec_sum); // Multiply and accumulate
        }

        // Handle the remaining elements
        double sum = 0.0;
        for (; j < end; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }

        // Store back the horizontal sum of vec_sum and add the remainder
        double sum_array[4];
        _mm256_storeu_pd(sum_array, vec_sum); // Store back to memory
        y[i] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum;
    }
}
*/
void spmv(const CSRMatrix* A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A->n_rows; i++) {
        __m512d vec_sum = _mm512_setzero_pd(); // Vectorized sum (8 doubles)

        int j = A->row_ptr[i];
        int end = A->row_ptr[i + 1];

        // Process in chunks of 8
        for (; j <= end - (end % 8); j += 8) {
            // Prefetch future values and indices
            __builtin_prefetch(&A->values[j + 4], 0, 1);
            __builtin_prefetch(&A->col_indices[j + 8], 0, 2);
	    __builtin_prefetch(&x[A->col_indices[j + 4]], 0, 1);
            // Load 8 values from the matrix
            __m512d vec_values = _mm512_load_pd(&A->values[j]);

            // Gather 8 elements of x using the column indices
            __m512d vec_x = _mm512_i32gather_pd(
                _mm256_loadu_si256((__m256i*)&A->col_indices[j]), // Indices as 8 32-bit integers
                x, // Base address
                8 // Scale factor (size of double)
            );

            // Multiply and accumulate
            vec_sum = _mm512_fmadd_pd(vec_values, vec_x, vec_sum);
        }

        // Handle the remaining elements
        double sum = 0.0;
        for (; j < end; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }

        // Reduce vec_sum to a scalar sum
        double horizontal_sum = _mm512_reduce_add_pd(vec_sum);

        // Store the result for row i
        y[i] = horizontal_sum + sum;
    }
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
    spmv(A, x, y);

    // Timing
    const int n_runs = 100;
    double start_time = get_time();
    
    for (int i = 0; i < n_runs; i++) {
        spmv(A, x, y);
    }
    
    double end_time = get_time();
    double elapsed_time = (end_time - start_time) / n_runs;

    // Calculate GFLOP/s
    // Each non-zero element requires 2 operations (multiply and add)
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
