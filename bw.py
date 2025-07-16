import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix

def compute_bandwidth_details(matrix):
    coo = matrix.tocoo()
    n_rows, n_cols = matrix.shape

    row_bw = np.zeros(n_rows, dtype=int)
    col_bw = np.zeros(n_cols, dtype=int)

    for i, j in zip(coo.row, coo.col):
        diff = abs(i - j)
        row_bw[i] = max(row_bw[i], diff)
        col_bw[j] = max(col_bw[j], diff)

    max_row_bw = np.max(row_bw)
    avg_row_bw = np.mean(row_bw)

    max_col_bw = np.max(col_bw)
    avg_col_bw = np.mean(col_bw)

    bandwidth = np.max(np.abs(coo.row - coo.col))
    return {
        "max_row_bandwidth": max_row_bw,
        "avg_row_bandwidth": avg_row_bw,
        "max_col_bandwidth": max_col_bw,
        "avg_col_bandwidth": avg_col_bw,
        "bandwidth" : bandwidth,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute detailed bandwidths of a sparse matrix.")
    parser.add_argument("filename", type=str, help="Path to Matrix Market (.mtx) file")
    args = parser.parse_args()

    matrix = mmread(args.filename)

    stats = compute_bandwidth_details(matrix)

    print(f"Matrix shape: {matrix.shape}")
    for k, v in stats.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v:.2f}")

