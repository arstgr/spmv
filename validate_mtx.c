#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double val;
} triple;

int comp(const void *a, const void *b) {
    triple *ta = (triple *)a;
    triple *tb = (triple *)b;
    if (ta->row != tb->row) {
        return ta->row - tb->row;
    }
    return ta->col - tb->col;
}

bool validate_matrix(triple *entries, int nnz, int rows, int cols) {
    for (int i = 0; i < nnz; i++) {
        if (entries[i].row < 1 || entries[i].row > rows ||
            entries[i].col < 1 || entries[i].col > cols) {
            fprintf(stderr, "Error: Index out of bounds at entry %d: (%d, %d)\n",
                    i, entries[i].row, entries[i].col);
            return false;
        }

        if (i > 0 && entries[i].row == entries[i - 1].row &&
            entries[i].col == entries[i - 1].col) {
            fprintf(stderr, "Error: Duplicate entry at (%d, %d)\n",
                    entries[i].row, entries[i].col);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    FILE *in = fopen(input_file, "r");
    if (!in) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_file);
        return 1;
    }

    char line[8192];
    while (fgets(line, sizeof(line), in)) {
        if (line[0] != '%') {
            break;
        }
    }


    int rows, cols, nnz;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz) != 3) {
    	fprintf(stderr, "Error: Invalid header format\n");
    	fclose(in);
    	return 1;
    }
    triple *entries = (triple *)malloc(nnz * sizeof(triple));
    if (!entries) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(in);
        return 1;
    }

    for (int i = 0; i < nnz; i++) {
        if (fscanf(in, "%d %d %lf", &entries[i].row, &entries[i].col, &entries[i].val) != 3) {
            fprintf(stderr, "Error: Invalid entry format at line %d\n", i + 2);
            free(entries);
            fclose(in);
            return 1;
        }
    }

    fclose(in);

    // Sort entries by row and column
    qsort(entries, nnz, sizeof(triple), comp);

    // Validate the matrix
    if (!validate_matrix(entries, nnz, rows, cols)) {
        free(entries);
        return 1;
    }

    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
        free(entries);
        return 1;
    }

    fprintf(out, "%d %d %d\n", rows, cols, nnz);
    for (int i = 0; i < nnz; i++) {
        fprintf(out, "%d %d %.17f\n", entries[i].row, entries[i].col, entries[i].val);
    }

    fclose(out);
    free(entries);

    printf("Matrix validated and written to %s\n", output_file);
    return 0;
}

