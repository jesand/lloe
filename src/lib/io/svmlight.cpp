/// @file  svmlight.cpp
/// @brief I/O tools for svmlight files.
///
/// See the documentation at http://svmlight.joachims.org.
/// To quote from their description of the file format:
///
/// The input file example_file contains the training examples. The first lines
/// may contain comments and are ignored if they start with #. Each of the
/// following lines represents one training example and is of the following
/// format:
///
/// <line> .=. <target> <feature>:<value> ... <feature>:<value> # <info>
/// <target> .=. +1 | -1 | 0 | <float> 
/// <feature> .=. <integer> | "qid"
/// <value> .=. <float>
/// <info> .=. <string>
///
/// The target value and each of the feature/value pairs are separated by a
/// space character. Feature/value pairs MUST be ordered by increasing feature
/// number. Features with value zero can be skipped. The string <info> can be
/// used to pass additional information to the kernel (e.g. non feature
/// vector data).

#include <ogt/io/io.hpp>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <boost/compute/detail/lru_cache.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#pragma GCC diagnostic pop

using boost::compute::detail::lru_cache;
using boost::phoenix::push_back;
using boost::spirit::ascii::space;
using boost::spirit::istream_iterator;
using boost::spirit::qi::_1;
using boost::spirit::qi::double_;
using boost::spirit::qi::eol;
using boost::spirit::qi::int_;
using boost::spirit::qi::phrase_parse;
using Eigen::SparseMatrix;
using std::atof;
using std::atoi;
using std::ifstream;
using std::istream;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

namespace OGT_NAMESPACE {
namespace io {

/// Allows row-by-row access to a very large sparse data file.
class SvmLightScanner : public SparseScanner {
public:

	/// Instantiate a new scanner from the specified file path
	SvmLightScanner(std::string name)
		: fin(openSource(name))
		, nRow(-1)
		, nCols(0)
		, maxRows(0)
		, gotEof(false)
		, cache(10000)
	{
	}

	/// Move to the beginning of the file.
	void reset() override {
		fin->clear();
		fin->seekg(0);
		nRow = -1;
	}

	/// Truncate the matrix to the specified number of rows.
	void truncate(size_t nRows) override {
		if (maxRows > 0 && nRows < maxRows) {
			maxRows = nRows;
		}
	}

	/// Get the (zero-based) index of the most recently-read row from the file.
	size_t nrow() override {
		return nRow;
	}

	/// Get the number of rows in the file.
	size_t rows() override {
		if (!gotEof) {
			const auto atRow = nRow;
			const auto pos = fin->tellg();
			while (!eof()) {
				read();
			}
			fin->clear();
			fin->seekg(pos);
			nRow = atRow;
		}
		return rowPos.size();
	}

	/// Get the number of columns in the file.
	size_t cols() override {
		if (!gotEof) {
			const auto pos = fin->tellg();
			while (!eof()) {
				read();
			}
			fin->clear();
			fin->seekg(pos);
		}
		return nCols;
	}

	/// Ask whether there is another vector in the file.
	bool eof() override {
		return fin->bad() || fin->eof();
	}

	/// Get the next vector from the file.
	SparseScanner::VT read() override {

		// Try to read the next line of input
		if ((maxRows > 0 && nRow >= static_cast<int>(maxRows - 1)) || eof()) {
			gotEof = true;
			return SparseScanner::VT();
		}
		string line;
		auto pos = fin->tellg();
		while (line == "") {
			getline(*fin, line);
			if (eof()) {
				gotEof = true;
				return SparseScanner::VT();
			} else if (line == "") {
				pos = fin->tellg();
			}
		}

		// Parse the row
		nRow++;
		if (static_cast<size_t>(nRow) >= rowPos.size()) {
			rowPos.push_back(pos);
		}

		bool first = true;
		typedef Eigen::Triplet<double> T;
		std::vector<T> tripletList;
		size_t spacePos = line.find(' ');
		while (spacePos != std::string::npos) {
			if (first) {
				// skip the target
				first = false;
				spacePos = line.find(' ', spacePos);
			} else {
				size_t colPos = line.find(':', spacePos + 1);
				if (colPos == std::string::npos) {
					spacePos = colPos;
				} else {
					string sidx = line.substr(spacePos + 1, colPos);
					if (sidx.find("qid") != string::npos) {
						spacePos = line.find(' ', colPos + 1);
						continue;
					}
					int idx = atoi(sidx.c_str());
					spacePos = line.find(' ', colPos + 1);
					double val = atof(line.substr(colPos + 1, spacePos).c_str());

					tripletList.emplace_back(0, idx, val);
					if (idx >= static_cast<int>(nCols)) {
						nCols = idx + 1;
					}
				}
			}
		}

		SparseScanner::VT vec(nCols);
		for (const auto& trip : tripletList) {
			vec.coeffRef(trip.col()) += trip.value();
		}
		return vec;
	}

	/// Get the specified vector from the file. May trigger a file scan.
	SparseScanner::VT row(size_t row) override {
		auto cached = cache.get(row);
		if (cached != boost::none) {
			return *cached;
		}
		if (row >= rowPos.size()) {
			while (!eof() && row >= rowPos.size()) {
				auto vec = read();
				if (row == static_cast<size_t>(nRow)) {
					cache.insert(row, vec);
					return vec;
				}
			}
			return SparseScanner::VT();
		} else {
			nRow = static_cast<int>(row) - 1;
			fin->clear();
			fin->seekg(rowPos[row]);
			auto vec = read();
			cache.insert(row, vec);
			return vec;
		}
	}

private:
	std::shared_ptr<istream> fin;
	int nRow;
	size_t nCols;
	size_t maxRows;
	vector<std::istream::pos_type> rowPos;
	bool gotEof;
	lru_cache<size_t, SparseScanner::VT> cache;
};

/// Get a scanner for a svmlight file
shared_ptr<SparseScanner> openSvmLightScanner(std::string name) {
	return std::make_shared<SvmLightScanner>(name);
}

} // end namespace io
} // end namespace OGT_NAMESPACE
