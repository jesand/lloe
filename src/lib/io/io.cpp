/// @file  io.cpp
/// @brief Shared I/O tools.

#include <ogt/io/io.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wconstant-conversion"
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#pragma GCC diagnostic pop

using boost::iostreams::bzip2_compressor;
using boost::iostreams::bzip2_decompressor;
using boost::iostreams::file_sink;
using boost::iostreams::file_source;
using boost::iostreams::filtering_stream;
using boost::iostreams::gzip_compressor;
using boost::iostreams::gzip_decompressor;
using boost::iostreams::input;
using boost::iostreams::output;
using boost::iostreams::zlib_compressor;
using boost::iostreams::zlib_decompressor;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using std::istream;
using std::ostream;
using std::shared_ptr;
using std::string;
using std::vector;

namespace OGT_NAMESPACE {
namespace io {

/// The type of compression to use for a stream
enum CompressionType {
	ct_none,
	ct_bzip2,
	ct_gzip,
	ct_zlib,
};

/// Choose the compression type for a source or sink.
CompressionType getCompressionType(string name) {
	if (boost::ends_with(name, ".gz") || boost::ends_with(name, ".gzip")) {
		return ct_gzip;
	} else if (boost::ends_with(name, ".bz2")) {
		return ct_bzip2;
	} else if (boost::ends_with(name, ".zip")) {
		return ct_zlib;
	} else {
		return ct_none;
	}
}

/// Opens a file for reading with on-the-fly decompression as needed.
std::unique_ptr<std::istream> openSource(string name) {
	std::unique_ptr<filtering_stream<input>>
		in(new filtering_stream<input>);

	// Note that inline decompression is incompatible with seeking (without
	// additional development work).
	switch (getCompressionType(name)) {
		case ct_none:
			break;
		case ct_bzip2:
			in->push(bzip2_decompressor());
			break;
		case ct_gzip:
			in->push(gzip_decompressor());
			break;
		case ct_zlib:
			in->push(zlib_decompressor());
			break;
	}
	in->push(file_source(name));
	return std::unique_ptr<std::istream>(in.release());
}

/// Opens a file for writing with on-the-fly compression as needed.
std::unique_ptr<std::ostream> openSink(string name) {
	std::unique_ptr<filtering_stream<output>> out(new filtering_stream<output>);
	switch (getCompressionType(name)) {
		case ct_none:
			break;
		case ct_bzip2:
			out->push(bzip2_compressor());
			break;
		case ct_gzip:
			out->push(gzip_compressor());
			break;
		case ct_zlib:
			out->push(zlib_compressor());
			break;
	}
	out->push(file_sink(name));
	return std::unique_ptr<std::ostream>(out.release());
}

/// Ask whether the given filename is for a sparse file format.
bool isSparseFile(string name) {
	return name.find(".svmlight") != string::npos;
}

/// Opens a scanner to read a matrix from the specified location.
shared_ptr<SparseScanner> openSparseScanner(string name) {
	if (name.find(".svmlight") != string::npos) {
		return openSvmLightScanner(name);
	} else {
		return shared_ptr<SparseScanner>();
	}
}

/// A Scanner for an in-memory dense matrix.
class DenseScannerImpl : public DenseScanner {
public:
	virtual ~DenseScannerImpl() = default;

	/// Constructor
	DenseScannerImpl(MatrixXd matrix)
		: matrix(matrix)
		, nRow(-1)
	{}

	/// Move to the beginning of the file.
	void reset() override {
		nRow = -1;
	}

	/// Truncate the matrix to the specified number of rows.
	void truncate(size_t nRows) override {
		if (nRows > static_cast<size_t>(matrix.rows())) {
			matrix.conservativeResize(nRows, matrix.cols());
			if (static_cast<size_t>(nRow) >= nRows) {
				nRow = nRows - 1;
			}
		}
	}

	/// Get the (zero-based) index of the most recently-read row from the file.
	size_t nrow() override {
		return static_cast<size_t>(nRow);
	}

	/// Get the number of columns in the file. May trigger a full file scan.
	size_t cols() override {
		return matrix.cols();
	}

	/// Get the number of rows in the file. May trigger a full file scan.
	size_t rows() override {
		return matrix.rows();
	}

	/// Ask whether there is another vector in the file.
	bool eof() override {
		return nRow >= matrix.rows() - 1;
	}

	/// Get the next vector from the file.
	VectorXd read() override {
		nRow++;
		if (nRow >= matrix.rows()) {
			return VectorXd();
		} else {
			return matrix.row(nRow);
		}
	}

	/// Get the specified vector from the file. May trigger a file scan.
	inline VectorXd row(size_t row) override {
		return matrix.row(row);
	}

private:
	MatrixXd matrix;
	Index nRow;
};

/// Opens a scanner to read a matrix from the specified location.
shared_ptr<DenseScanner> openDenseScanner(string name, string
#if (HAVE_MATLAB)
	matvar
#endif
	) {
#if (HAVE_MATLAB)
	if (matvar != "") {
		auto matrix = loadMatlabMatrix(name, matvar);
		return std::make_shared<DenseScannerImpl>(matrix);
	}
#endif

	auto matrix = loadCsvMatrix(name);
	return std::make_shared<DenseScannerImpl>(matrix);
}

/// Load a full matrix from a sparse file
SparseMatrix<double, RowMajor> loadSparseMatrix(string name) {
	auto scanner = openSparseScanner(name);

	vector<Triplet<double>> triplets;
	for (Index ii = 0; !scanner->eof(); ii++) {
		auto vec = scanner->read();
		for (SparseScanner::VT::InnerIterator it(vec); it; ++it) {
			triplets.emplace_back(ii, it.index(), it.value());
		}
	}

	SparseMatrix<double, RowMajor> matrix(scanner->rows(), scanner->cols());
	matrix.setFromTriplets(triplets.begin(), triplets.end());
	return matrix;
}

/// Load a subset of a matrix from a sparse file
SparseMatrix<double, RowMajor> loadSparseMatrix(const VectorXd& objects, string name) {
	auto scanner = openSparseScanner(name);
	vector<Triplet<double>> triplets;
	for (Index ii = 0; ii < objects.size(); ii++) {
		auto vec = scanner->row(objects[ii]);
		for (SparseScanner::VT::InnerIterator it(vec); it; ++it) {
			triplets.emplace_back(ii, it.index(), it.value());
		}
	}
	SparseMatrix<double, RowMajor> matrix(objects.size(), scanner->cols());
	matrix.setFromTriplets(triplets.begin(), triplets.end());
	return matrix;
}

/// Load a full matrix from a dense file
MatrixXd loadDenseMatrix(string name, string matvar) {
	auto scanner = openDenseScanner(name, matvar);
	MatrixXd matrix(scanner->rows(), scanner->cols());
	for (Index ii = 0; ii < matrix.rows(); ii++) {
		matrix.row(ii) = scanner->row(ii);
	}
	return matrix;
}

/// Load a subset of a matrix from a dense file
MatrixXd loadDenseMatrix(const VectorXd& objects, string name, string matvar) {
	auto scanner = openDenseScanner(name, matvar);
	MatrixXd matrix(objects.size(), scanner->cols());
	for (Index ii = 0; ii < matrix.rows(); ii++) {
		matrix.row(ii) = scanner->row(objects[ii]);
	}
	return matrix;
}

} // end namespace io
} // end namespace OGT_NAMESPACE
