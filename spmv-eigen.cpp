#include <iostream>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>
#include <sstream>
#include <omp.h>
#include <chrono>

#define EIGEN_USE_THREADS
#define EIGEN_DONT_PARALLELIZE

void readMatrixMarket(const std::string& filename, Eigen::SparseMatrix<double, Eigen::RowMajor>& sparseMatrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::vector<Eigen::Triplet<double>> triplets;
    int rows = 0, cols = 0, nonZeros = 0;

    while (std::getline(file, line)) {
        if (line[0] == '%') continue; // Skip comments
        std::istringstream iss(line);
        if (!(iss >> rows >> cols >> nonZeros)) {
            throw std::runtime_error("Invalid header line in Matrix Market file.");
        }
        break;
    }

    int row, col;
    double value;
    int count = 0; // Count non-zero entries read
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> row >> col >> value)) {
            throw std::runtime_error("Error reading a matrix entry. Ensure the file is well-formatted.");
        }

        // Convert to zero-based indexing
        triplets.emplace_back(row - 1, col - 1, value);
        count++;
    }

    file.close();

    if (count != nonZeros) {
        throw std::runtime_error("Mismatch between declared non-zeros and actual entries read.");
    }

    sparseMatrix.resize(rows, cols);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_matrix_file>" << std::endl;
        return 1;
    }
    std::string matrixFile = argv[1];

    // note" rowmajor to have CSR and use multithreading"
    // https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparseMatrix;

    try {
        readMatrixMarket(matrixFile, sparseMatrix);
	int rows = sparseMatrix.rows();
	int cols = sparseMatrix.cols();
	int nonZeros = sparseMatrix.nonZeros();

	std::cout << "Sparse Matrix Read:" << rows << " rows, " << cols << " cols, " << nonZeros << " nnzs." << std::endl;

        //std::cout << "Sparse Matrix (CSR format):\n" << Eigen::MatrixXd(sparseMatrix) << "\n";

    	#pragma omp parallel
    	{
    	    #pragma omp single
    	    std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
    	}


        Eigen::VectorXd denseVector(cols);
	denseVector.setOnes();

	auto start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd result = sparseMatrix * denseVector;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::micro> elapsed = end - start;
	double gflops = (2.0 * nonZeros * 1e-9) / (elapsed.count() * 1e-6);
	std::cout << "Sparse Matrix-Vector Multiplication Runtime: " << elapsed.count() << " us" << std::endl;
	std::cout << gflops << "GFLOPS" << std::endl;

        //std::cout << "Dense Vector:\n" << denseVector << "\n\n";
        //std::cout << "Result (Sparse Matrix * Dense Vector):\n" << result << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

