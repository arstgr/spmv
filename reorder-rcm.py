import argparse
from scipy.io import mmread, mmwrite
from scipy.sparse import issparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np

def matrix_bandwidth(A):
    """Compute the bandwidth of a sparse matrix A."""
    Acoo = A.tocoo()
    if Acoo.nnz == 0:
        return 0
    bandwidth = np.max(np.abs(Acoo.row - Acoo.col))
    return bandwidth

def main():
    parser = argparse.ArgumentParser(description="Read a matrix market file, reorder using RCM, and write to another file.")
    parser.add_argument("input_file", help="Input Matrix Market file (.mtx)")
    parser.add_argument("output_file", help="Output Matrix Market file (.mtx)")
    args = parser.parse_args()

    # Read the matrix
    print(f"Reading matrix from {args.input_file} ...")
    A = mmread(args.input_file)

    # Ensure it's in sparse format
    if not issparse(A):
        print("Input is not a sparse matrix, converting to CSR format.")
        from scipy.sparse import csr_matrix
        A = csr_matrix(A)
    else:
        A = A.tocsr()

    # Calculate bandwidth before reordering
    bw_before = matrix_bandwidth(A)
    print(f"Matrix bandwidth before RCM: {bw_before}")

    # Compute the RCM permutation
    print("Computing Reverse Cuthill-McKee ordering ...")
    perm = reverse_cuthill_mckee(A)

    # Apply the permutation
    print("Reordering matrix ...")
    A_rcm = A[perm, :][:, perm]

    # Calculate bandwidth after reordering
    bw_after = matrix_bandwidth(A_rcm)
    print(f"Matrix bandwidth after RCM: {bw_after}")

    # Write the reordered matrix
    print(f"Writing reordered matrix to {args.output_file} ...")
    #mmwrite(args.output_file, A_rcm)
    mmwrite(args.output_file, A_rcm.tocoo())
    print("Done.")

if __name__ == "__main__":
    main()
