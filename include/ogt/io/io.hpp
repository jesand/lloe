/// @file  io.hpp
/// @brief Declarations related to low-level data I/O.

#pragma once
#ifndef OGT_IO_IO_HPP
#define OGT_IO_IO_HPP

#include <ogt/config.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace OGT_NAMESPACE {
namespace io {

/// The type thrown on I/O errors.
struct IoErr : public std::runtime_error {
	virtual ~IoErr() = default;

	/// Build an IoErr with the specified message.
	IoErr(std::string message)
		: std::runtime_error("ogt::IoErr: " + message) {
	}
};

/// Opens a file for reading with on-the-fly decompression as needed.
/// Decompression is triggered by the file extensions .zip, .gz, .gzip, or .bz2.
std::unique_ptr<std::istream> openSource(std::string name);

/// Opens a file for writing with on-the-fly compression as needed.
/// Compression is triggered by the file extensions .zip, .gz, .gzip, or .bz2.
std::unique_ptr<std::ostream> openSink(std::string name);

/// Loads a matrix from a .csv file from the specified stream. Does not close
/// the stream when complete. The name parameter is used to identify the stream
/// in error messages.
/// The specified number of initial rows and columns will be ignored.
/// Throws IoErr on failure.
Eigen::MatrixXd loadCsvMatrix(std::string name, std::istream& source,
	char fieldSep=',', char rowSep='\n', size_t skipRows=0, size_t skipCols=0);

/// Loads a matrix from a .csv file at the specified path.
/// The specified number of initial rows and columns will be ignored.
/// Throws IoErr on failure.
Eigen::MatrixXd loadCsvMatrix(std::string file, char fieldSep=',',
	char rowSep='\n', size_t skipRows=0, size_t skipCols=0);

/// Saves a matrix as .csv to the specified stream. Does not close
/// the stream when complete. The name parameter is used to identify the stream
/// in error messages.
/// Throws IoErr on failure.
void saveCsvMatrix(const Eigen::MatrixXd& matrix, std::string name,
	std::ostream& dest, char fieldSep=',', char rowSep='\n');

/// Saves a matrix to a .csv file at the specified path.
/// Throws IoErr on failure.
void saveCsvMatrix(const Eigen::MatrixXd& matrix, std::string file,
	char fieldSep=',', char rowSep='\n');

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, std::string name,
	std::ostream& dest);

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, std::string file);

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding,
	const std::vector<size_t>& ids, std::string name, std::ostream& dest);

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding,
	const std::vector<size_t>& ids, std::string file);

#if (HAVE_MATLAB)
/// Loads a matrix from a .mat file or throws an IoErr.
Eigen::MatrixXd loadMatlabMatrix(std::string file, std::string var);
#endif

/// Base class for row-by-row access to matrices.
template<typename vector_type>
struct Scanner {
	virtual ~Scanner() = default;

	/// Declare the vector type for external reference.
	typedef vector_type VT;

	/// Move to the beginning of the file.
	virtual void reset() = 0;

	/// Get the (zero-based) index of the most recently-read row from the file.
	virtual size_t nrow() = 0;

	/// Get the number of columns in the file. May trigger a full file scan.
	virtual size_t cols() = 0;

	/// Get the number of rows in the file. May trigger a full file scan.
	virtual size_t rows() = 0;

	/// Truncate to the specified number of rows.
	virtual void truncate(size_t nRows) = 0;

	/// Ask whether there is another vector in the file.
	virtual bool eof() = 0;

	/// Get the next vector from the file.
	virtual vector_type read() = 0;

	/// Get the specified vector from the file. May trigger a file scan.
	virtual vector_type row(size_t row) = 0;
};

/// Allows row-by-row access to a sparse matrix.
typedef Scanner<Eigen::SparseVector<double>> SparseScanner;

/// Allows row-by-row access to a dense matrix.
typedef Scanner<Eigen::VectorXd> DenseScanner;

/// Get a scanner for a svmlight file
std::shared_ptr<SparseScanner> openSvmLightScanner(std::string name);

/// Save a sparse binary file which more efficiently stores sparse data
void saveSpBinMatrix(std::shared_ptr<SparseScanner> matrix, std::string file);

/// Open a scanner for a sparse binary matrix.
std::shared_ptr<SparseScanner> openSpBinScanner(std::string name);

/// Ask whether the given filename is for a sparse file format.
bool isSparseFile(std::string name);

/// Opens a scanner to read a matrix from the specified location.
std::shared_ptr<SparseScanner> openSparseScanner(std::string name);

/// Load a full matrix from a sparse file
Eigen::SparseMatrix<double, Eigen::RowMajor> loadSparseMatrix(std::string name);

/// Load a subset of a matrix from a sparse file
Eigen::SparseMatrix<double, Eigen::RowMajor> loadSparseMatrix(const Eigen::VectorXd& objects,
	std::string name);

/// Opens a scanner to read a matrix from the specified location.
std::shared_ptr<DenseScanner> openDenseScanner(std::string name,
	std::string matvar = "");

/// Load a full matrix from a dense file
Eigen::MatrixXd loadDenseMatrix(std::string name, std::string matvar = "");

/// Load a subset of a matrix from a dense file
Eigen::MatrixXd loadDenseMatrix(const Eigen::VectorXd& objects,
	std::string name, std::string matvar = "");

} // end namespace io
} // end namespace OGT_NAMESPACE
#endif /* OGT_IO_IO_HPP */
