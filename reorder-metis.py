import argparse
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import issparse, csr_matrix

import pymetis

def to_adjacency_list(A):
    """Create adjacency list for METIS from a symmetric sparse matrix."""
    n = A.shape[0]
    A = A.tocoo()
    adj = [[] for _ in range(n)]
    for i, j in zip(A.row, A.col):
        if i != j:
            adj[i].append(j)
            adj[j].append(i)
    # Remove duplicates
    adj = [list(set(neigh)) for neigh in adj]
    return adj

def matrix_bandwidth(A):
    """Compute the bandwidth of a sparse matrix A."""
    Acoo = A.tocoo()
    if Acoo.nnz == 0:
        return 0
    return np.max(np.abs(Acoo.row - Acoo.col))

def main():
    parser = argparse.ArgumentParser(description="Reorder a matrix using METIS nested dissection for SpMV locality.")
    parser.add_argument("input_file", help="Input Matrix Market file (.mtx)")
    parser.add_argument("output_file", help="Output Matrix Market file (.mtx)")
    args = parser.parse_args()

    print(f"Reading matrix from {args.input_file} ...")
    A = mmread(args.input_file)
    if not issparse(A):
        print("Input is not a sparse matrix, converting to CSR format.")
        A = csr_matrix(A)
    else:
        A = A.tocsr()

    # Calculate bandwidth before reordering
    bw_before = matrix_bandwidth(A)
    print(f"Matrix bandwidth before METIS: {bw_before}")

    # For METIS, the matrix should be symmetric. If not, symmetrize it.
    if (A != A.transpose()).nnz != 0:
        print("Matrix is not symmetric, symmetrizing for METIS ordering.")
        A_sym = ((A + A.transpose()) > 0).astype(int)
    else:
        A_sym = ((A + A.transpose()) > 0).astype(int)

    print("Building adjacency list for METIS ...")
    adj = to_adjacency_list(A_sym)

    print("Calling PyMetis for nested dissection ordering ...")
    perm, _ = pymetis.nested_dissection(adj)
    print("Permutation computed.")

    print("Reordering matrix ...")
    A_perm = A[perm, :][:, perm]

    # Calculate bandwidth after reordering
    bw_after = matrix_bandwidth(A_perm)
    print(f"Matrix bandwidth after METIS: {bw_after}")

    print(f"Writing reordered matrix to {args.output_file} ...")
    mmwrite(args.output_file, A_perm.tocoo())
    print("Done.")

if __name__ == "__main__":
    main()
