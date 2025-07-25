#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

void* posix_aligned_alloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0)
        ptr = NULL;
    return ptr;
}

void read_mm_header(FILE* f, int* M, int* N, int* nz) {
    char line[1024];
    do { fgets(line, 1024, f); } while (line[0] == '%');
    sscanf(line, "%d %d %d", M, N, nz);
}
void read_mm_coo(FILE* f, int nz, int* I, int* J, double* val) {
    for (int k = 0; k < nz; ++k) {
        int i, j;
        double v;
        fscanf(f, "%d %d %lf", &i, &j, &v);
        I[k] = i - 1;
        J[k] = j - 1;
        val[k] = v;
    }
}

void coo_to_csr(int M, int nz, int* I, int* J, double* val,
                int** row_ptr, int** col_idx, double** csr_val) {
    *row_ptr = (int*)posix_aligned_alloc((M + 1) * sizeof(int), 64);
    *col_idx = (int*)posix_aligned_alloc(nz * sizeof(int), 64);
    *csr_val = (double*)posix_aligned_alloc(nz * sizeof(double), 64);
    if (!*row_ptr || !*col_idx || !*csr_val) {
        fprintf(stderr, "CSR aligned memory allocation failed\n");
        exit(1);
    }
    memset(*row_ptr, 0, (M + 1) * sizeof(int));
    for (int k = 0; k < nz; ++k) (*row_ptr)[I[k] + 1]++;
    for (int i = 0; i < M; ++i) (*row_ptr)[i + 1] += (*row_ptr)[i];
    int* next = (int*)calloc(M, sizeof(int));
    for (int k = 0; k < nz; ++k) {
        int row = I[k];
        int dest = (*row_ptr)[row] + next[row];
        (*col_idx)[dest] = J[k];
        (*csr_val)[dest] = val[k];
        next[row]++;
    }
    free(next);
}

void csr_spmv(const int M, const int* row_ptr, const int* col_idx, const double* csr_val,
              const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int jj = row_ptr[i]; jj < row_ptr[i + 1]; ++jj)
            sum += csr_val[jj] * x[col_idx[jj]];
        y[i] = sum;
    }
}

void coo_to_ellpack(int M, int nz, int* I, int* J, double* val,
                    double** ell_val, int** ell_idx, int* max_nnz_per_row) {
    int* row_counts = (int*)calloc(M, sizeof(int));
    for (int k = 0; k < nz; ++k) row_counts[I[k]]++;
    int max_nnz = 0;
    for (int i = 0; i < M; ++i)
        if (row_counts[i] > max_nnz) max_nnz = row_counts[i];
    *max_nnz_per_row = max_nnz;
    free(row_counts);

    size_t arr_size = M * max_nnz;
    *ell_val = (double*)posix_aligned_alloc(arr_size * sizeof(double), 64);
    *ell_idx = (int*)posix_aligned_alloc(arr_size * sizeof(int), 64);
    if (!*ell_val || !*ell_idx) {
        fprintf(stderr, "ELLPACK aligned memory allocation failed\n");
        exit(1);
    }
    memset(*ell_val, 0, arr_size * sizeof(double));
    for (size_t i = 0; i < arr_size; ++i) (*ell_idx)[i] = -1;

    int* offset = (int*)calloc(M, sizeof(int));
    for (int k = 0; k < nz; ++k) {
        int row = I[k];
        int pos = offset[row]++;
        (*ell_val)[row * max_nnz + pos] = val[k];
        (*ell_idx)[row * max_nnz + pos] = J[k];
    }
    free(offset);
}

void ellpack_spmv(const int n_rows, const int max_nnz_per_row, const double* val, const int* indices, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        int base = i * max_nnz_per_row;
        for (int j = 0; j < max_nnz_per_row; ++j) {
            int col = indices[base + j];
            if (col >= 0) {
                sum += val[base + j] * x[col];
            }
        }
        y[i] = sum;
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double l2_error(int n, const double* a, const double* b) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s matrix.mtx\n", argv[0]);
        return 1;
    }
    const char* mm_file = argv[1];

    FILE* f = fopen(mm_file, "r");
    if (!f) { perror("Matrix Market file"); return 1; }
    int M, N, nz;
    read_mm_header(f, &M, &N, &nz);

    int* I = (int*)posix_aligned_alloc(nz * sizeof(int), 64);
    int* J = (int*)posix_aligned_alloc(nz * sizeof(int), 64);
    double* val = (double*)posix_aligned_alloc(nz * sizeof(double), 64);
    if (!I || !J || !val) {
        fprintf(stderr, "COO aligned memory allocation failed\n");
        exit(1);
    }
    read_mm_coo(f, nz, I, J, val);
    fclose(f);

    int *row_ptr, *col_idx;
    double *csr_val;
    coo_to_csr(M, nz, I, J, val, &row_ptr, &col_idx, &csr_val);

    double* x = (double*)posix_aligned_alloc(N * sizeof(double), 64);
    double* y_csr = (double*)posix_aligned_alloc(M * sizeof(double), 64);
    if (!x || !y_csr) {
        fprintf(stderr, "Vector aligned memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < N; ++i) x[i] = 1.0;

    const int n_runs = 100;
    double start_time = get_time();
    for (int i = 0; i < n_runs; i++) {
        csr_spmv(M, row_ptr, col_idx, csr_val, x, y_csr);
    }
    double end_time = get_time();
    double elapsed_time = (end_time - start_time) / n_runs;
    double gflops = (2.0 * nz * 1e-9) / elapsed_time;
    printf("Matrix dimensions: %d x %d\n", M, N);
    printf("Number of non-zeros: %d\n", nz);
    printf("Average time per SpMV (CSR): %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);
    
    double *ell_val;
    int *ell_idx;
    int max_nnz_per_row;
    coo_to_ellpack(M, nz, I, J, val, &ell_val, &ell_idx, &max_nnz_per_row);

    double* y_ellpack = (double*)posix_aligned_alloc(M * sizeof(double), 64);
    if (!y_ellpack) {
        fprintf(stderr, "ELLPACK result aligned memory allocation failed\n");
        exit(1);
    }
    ellpack_spmv(M, max_nnz_per_row, ell_val, ell_idx, x, y_ellpack);

    start_time = get_time();
    for (int i = 0; i < n_runs; i++) {
        ellpack_spmv(M, max_nnz_per_row, ell_val, ell_idx, x, y_ellpack);
    }
    end_time = get_time();
    elapsed_time = (end_time - start_time) / n_runs;
    gflops = (2.0 * nz * 1e-9) / elapsed_time;
    printf("Matrix dimensions: %d x %d\n", M, N);
    printf("Number of non-zeros: %d\n", nz);
    printf("Average time per SpMV (Ellpack): %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);

    double error = l2_error(M, y_csr, y_ellpack);
    printf("L2 error between y_csr and y_ellpack: %.12g\n", error);

    //for (int i=0; i<M; i++)
    //    printf("y_csr[%d]=%f y_ellpack[%d]=%f\n",i,y_csr[i], i, y_ellpack[i]);

    free(I); free(J); free(val);
    free(row_ptr); free(col_idx); free(csr_val);
    free(x); free(y_csr); free(ell_val); free(ell_idx); free(y_ellpack);
    return 0;
}

