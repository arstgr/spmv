#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>

#define ALIGNMENT 64
#define SLICE_SIZE 64 // C (number of rows per slice)

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
    *row_ptr = (int*)posix_aligned_alloc((M + 1) * sizeof(int), ALIGNMENT);
    *col_idx = (int*)posix_aligned_alloc(nz * sizeof(int), ALIGNMENT);
    *csr_val = (double*)posix_aligned_alloc(nz * sizeof(double), ALIGNMENT);
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

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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

typedef struct {
    int num_slices;         // Number of slices
    int slice_size;         // C (rows per slice)
    int* slice_lengths;     // Array of max nnz per slice (σ)
    double* val;            // Values, contiguous for all slices
    int* col_idx;           // Indices, contiguous for all slices
} SELLCSigma;

static int max_in_range(const int* arr, int start, int end) {
    int max = 0;
    for (int i = start; i < end; ++i)
        if (arr[i] > max) max = arr[i];
    return max;
}

SELLCSigma coo_to_sellcsigma(int M, int nz, int* I, int* J, double* val, int slice_size) {
    int num_slices = (M + slice_size - 1) / slice_size;
    int* row_counts = calloc(M, sizeof(int));
    int** row_ptrs = malloc(M * sizeof(int*)); // Array of pointers to entry indices for each row
    int* row_pos = calloc(M, sizeof(int));     // Position counters per row

    // Count nnz per row
    for (int k = 0; k < nz; ++k) row_counts[I[k]]++;

    // Build row_ptrs: for each row, allocate space for indices
    for (int i = 0; i < M; ++i)
        row_ptrs[i] = malloc(row_counts[i] * sizeof(int));

    // Populate row_ptrs with indices into COO arrays
    int* row_fill = calloc(M, sizeof(int));
    for (int k = 0; k < nz; ++k) {
        int row = I[k];
        row_ptrs[row][row_fill[row]++] = k;
    }
    free(row_fill);

    // Compute slice_lengths and total_nnz
    int* slice_lengths = malloc(num_slices * sizeof(int));
    int total_nnz = 0;
    for (int s = 0; s < num_slices; ++s) {
        int slice_start = s * slice_size;
        int slice_end = (slice_start + slice_size > M) ? M : (slice_start + slice_size);
        int maxlen = 0;
        for (int r = slice_start; r < slice_end; ++r)
            if (row_counts[r] > maxlen) maxlen = row_counts[r];
        slice_lengths[s] = maxlen;
        total_nnz += (slice_end - slice_start) * maxlen;
    }

    double* sell_val = posix_aligned_alloc(total_nnz * sizeof(double), ALIGNMENT);
    int* sell_idx = posix_aligned_alloc(total_nnz * sizeof(int), ALIGNMENT);
    memset(sell_val, 0, total_nnz * sizeof(double));
    for (size_t i = 0; i < total_nnz; ++i) sell_idx[i] = 0; // pad with 0 for vectorization

    int pos = 0;
    for (int s = 0; s < num_slices; ++s) {
        int slice_start = s * slice_size;
        int slice_end = (slice_start + slice_size > M) ? M : (slice_start + slice_size);
        int slice_len = slice_lengths[s];
        for (int r = slice_start; r < slice_end; ++r) {
            int row_nnz = row_counts[r];
            for (int j = 0; j < slice_len; ++j) {
                int idx = pos++;
                if (j < row_nnz) {
                    int k = row_ptrs[r][j];
                    sell_val[idx] = val[k];
                    sell_idx[idx] = J[k];
                }
                // else already padded with 0
            }
        }
    }

    for (int i = 0; i < M; ++i) free(row_ptrs[i]);
    free(row_ptrs);
    free(row_counts);

    SELLCSigma sellc;
    sellc.num_slices = num_slices;
    sellc.slice_size = slice_size;
    sellc.slice_lengths = slice_lengths;
    sellc.val = sell_val;
    sellc.col_idx = sell_idx;
    return sellc;
}

void sellcsigma_spmv(const SELLCSigma* S, const int M, const double* x, double* y) {
    #pragma omp parallel for
    for (int s = 0; s < S->num_slices; ++s) {
        int slice_start = s * S->slice_size;
        int slice_end = (slice_start + S->slice_size > M) ? M : (slice_start + S->slice_size);
        int slice_len = S->slice_lengths[s];
        int base = 0;
        for (int i = 0; i < s; ++i) base += S->slice_lengths[i] * S->slice_size;
        for (int r = slice_start; r < slice_end; ++r) {
            double sum = 0.0;
            int row_base = base + (r - slice_start) * slice_len;
            for (int j = 0; j < slice_len; ++j) {
                sum += S->val[row_base + j] * x[S->col_idx[row_base + j]];
            }
            y[r] = sum;
        }
    }
}

// --- L2 norm of difference ---
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
    int M = 0, N = 0, nz = 0;
    read_mm_header(f, &M, &N, &nz);
    if (M <= 0 || N <= 0 || nz <= 0) {
        fprintf(stderr, "Invalid Matrix Market header: M=%d N=%d nz=%d\n", M, N, nz);
        exit(1);
    }

    int* I = (int*)posix_aligned_alloc(nz * sizeof(int), ALIGNMENT);
    int* J = (int*)posix_aligned_alloc(nz * sizeof(int), ALIGNMENT);
    double* val = (double*)posix_aligned_alloc(nz * sizeof(double), ALIGNMENT);
    if (!I || !J || !val) {
        fprintf(stderr, "COO aligned memory allocation failed\n");
        exit(1);
    }
    read_mm_coo(f, nz, I, J, val);
    fclose(f);

    int *row_ptr = NULL, *col_idx = NULL;
    double *csr_val = NULL;
    coo_to_csr(M, nz, I, J, val, &row_ptr, &col_idx, &csr_val);

    double* x = (double*)posix_aligned_alloc(N * sizeof(double), ALIGNMENT);
    double* y_csr = (double*)posix_aligned_alloc(M * sizeof(double), ALIGNMENT);
    if (!x || !y_csr) {
        fprintf(stderr, "Vector aligned memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < N; ++i) x[i] = 1.0;
    
    int n_runs = 100;
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

    printf("SpMV using SELL-C-sigma\n");
    SELLCSigma sellc = coo_to_sellcsigma(M, nz, I, J, val, SLICE_SIZE);

    printf("conversion finished\n");
    double* y_sellc = (double*)posix_aligned_alloc(M * sizeof(double), ALIGNMENT);
    if (!y_sellc) {
        fprintf(stderr, "SELL-C-sigma result aligned memory allocation failed\n");
        exit(1);
    }

    n_runs = 100;
    start_time = get_time();
    for (int i = 0; i < n_runs; i++) {
        sellcsigma_spmv(&sellc, M, x, y_sellc);
    }
    end_time = get_time();
    elapsed_time = (end_time - start_time) / n_runs;
    gflops = (2.0 * nz * 1e-9) / elapsed_time;
    printf("Matrix dimensions: %d x %d\n", M, N);
    printf("Number of non-zeros: %d\n", nz);
    printf("Average time per SpMV (SELL-C-sigma): %f seconds\n", elapsed_time);
    printf("Performance: %f GFLOP/s\n", gflops);

    double error = l2_error(M, y_csr, y_sellc);
    printf("L2 error between y_csr and SELL-C-sigma: %.12g\n", error);

    //for (int i=0; i<M; i++)
    //    printf("y_csr[%d]=%f y_ellpack[%d]=%f\n",i,y_csr[i], i, y_ellpack[i]);

    free(I); free(J); free(val);
    free(row_ptr); free(col_idx); free(csr_val);
    free(x); free(y_csr);
    free(sellc.slice_lengths); free(sellc.val); free(sellc.col_idx); free(y_sellc);
    return 0;
}
