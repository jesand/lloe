/// @file  csv.cpp
/// @brief I/O tools for CSV files.

#include <ogt/io/io.hpp>
#include <array>
#include <fstream>
#include <sstream>
#include <vector>

using std::array;
using std::atof;
using std::endl;
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;
using std::string;
using std::stringstream;
using std::vector;

namespace OGT_NAMESPACE {
namespace io {

/// Loads a matrix from a .csv file from the specified stream.
Eigen::MatrixXd loadCsvMatrix(string name, istream& source,
	char fieldSep, char rowSep, size_t skipRows, size_t skipCols) {
	const size_t MAXROW = 1024*1024, MAXFLD = 1024;
	array<char,MAXROW> row;
	array<char,MAXFLD> field;
	size_t lineNum = 0;
	size_t nCols = 0;
	vector<vector<double>> data;
	while (source.getline(row.data(), MAXROW, rowSep)) {
		if (++lineNum >= skipRows) {
			stringstream ssRow(row.data());
			size_t fieldNum = 0;
			vector<double> rdata;
			if (nCols > 0) {
				rdata.reserve(nCols);
			}
			while (ssRow.getline(field.data(), MAXFLD, fieldSep)) {
				if (++fieldNum >= skipCols) {
					stringstream ssField(field.data());
					double val;
					if (!(ssField >> val)) {
						throw IoErr("CSV error: invalid double format in line "
							+ std::to_string(lineNum) + " of " + name);
					}
					rdata.push_back(val);
					if (nCols > 0 && rdata.size() > nCols) {
						throw IoErr("CSV error: too many columns in line "
							+ std::to_string(lineNum) + " of " + name);
					}
				}
			}
			if (nCols == 0) {
				nCols = rdata.size();
				if (nCols == 0) {
					throw IoErr("CSV error: no columns in " + name);
				}
			}
			data.push_back(rdata);
		}
	}
	if (data.size() == 0) {
		throw IoErr("CSV error: no rows in " + name);
	}

	Eigen::MatrixXd matrix(data.size(), nCols);
	for (Eigen::Index i = 0; i < matrix.rows(); i++) {
		for (Eigen::Index j = 0; j < matrix.cols(); j++) {
			matrix(i,j) = data[i][j];
		}
	}
	return matrix;
}

/// Loads a matrix from a .csv file at the specified path.
/// The specified number of initial rows and columns will be ignored.
Eigen::MatrixXd loadCsvMatrix(string file, char fieldSep,
	char rowSep, size_t skipRows, size_t skipCols) {
	auto source = openSource(file);
	if (!source || !*source) {
		throw IoErr("CSV error: Could not open file: " + file);
	}
	return loadCsvMatrix(file, *source, fieldSep, rowSep, skipRows, skipCols);
}

/// Saves a matrix as .csv to the specified stream. Does not close
/// the stream when complete. The name parameter is used to identify the stream
/// in error messages.
/// Throws IoErr on failure.
void saveCsvMatrix(const Eigen::MatrixXd& matrix, string name, ostream& dest,
	char fieldSep, char rowSep) {

	std::ios_base::iostate oldmask = dest.exceptions();
	dest.exceptions(std::ofstream::failbit);
	try {
		for (Eigen::Index mi = 0; mi < matrix.rows(); mi++) {
			for (Eigen::Index mj = 0; mj < matrix.cols(); mj++) {
				if (mj > 0) {
					dest << fieldSep << matrix(mi,mj);
				} else {
					dest << matrix(mi,mj);
				}
			}
			dest << rowSep;
		}
	} catch (std::ios_base::failure &err) {
		dest.exceptions(oldmask);
		throw IoErr("CSV error: Could not write " + name + ": " + err.what());
	}
	dest.exceptions(oldmask);
}

/// Saves a matrix to a .csv file at the specified path.
/// Throws IoErr on failure.
void saveCsvMatrix(const Eigen::MatrixXd& matrix, string file,
	char fieldSep, char rowSep) {
	auto sink = openSink(file);
	if (!sink || !*sink) {
		throw IoErr("CSV error: Could not open file: " + file);
	}
	return saveCsvMatrix(matrix, file, *sink, fieldSep, rowSep);
}

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, const vector<size_t>& ids,
	string name, ostream& dest) {
	if (embedding.rows() == 0 || embedding.cols() == 0) {
		throw IoErr("Attempted to save an empty embedding to " + name);
	}
	std::ios_base::iostate oldmask = dest.exceptions();
	dest.exceptions(std::ofstream::failbit);
	dest.precision(20);
	try {
		for (Eigen::Index mi = 0; mi < embedding.rows(); mi++) {
			const auto& row = embedding.row(mi);
			if (row.array().isNaN().any() || row.array().isInf().any()) {
				throw IoErr("Embedding contained non-finite values");
			}
			if (ids.empty()) {
				dest << mi;
			} else {
				dest << ids[mi];
			}
			for (Eigen::Index mj = 0; mj < embedding.cols(); mj++) {
				dest << ',' << row(mj);
			}
			dest << endl;
		}
	} catch (std::ios_base::failure &err) {
		dest.exceptions(oldmask);
		throw IoErr("CSV error: Could not write " + name + ": " + err.what());
	}
	dest.exceptions(oldmask);
}

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, const vector<size_t>& ids,
	string file) {
	auto sink = openSink(file);
	if (!sink || !*sink) {
		throw IoErr("CSV error: Could not open file: " + file);
	}
	saveEmbedding(embedding, ids, file, *sink);
}

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, std::string name,
	std::ostream& dest) {
	saveEmbedding(embedding, {}, name, dest);
}

/// Saves an embedding to a .csv file. Each row has an object identifier and
/// then its position. Rows with any NaN or Inf values will be skipped.
void saveEmbedding(const Eigen::MatrixXd& embedding, std::string file) {
	auto sink = openSink(file);
	if (!sink || !*sink) {
		throw IoErr("CSV error: Could not open file: " + file);
	}
	saveEmbedding(embedding, {}, file, *sink);
}

} // end namespace io
} // end namespace OGT_NAMESPACE
