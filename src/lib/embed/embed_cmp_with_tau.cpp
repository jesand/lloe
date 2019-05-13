/// @file  embed_cmp_with_tau.cpp
/// @brief Implements kernel methods based on Kendall's tau.

#include <ogt/embed/cmp.hpp>
#include <ogt/linalg/linalg.hpp>
#include <Eigen/Sparse>

using std::min;
using std::max;
using std::vector;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using OGT_NAMESPACE::linalg::kernelForFeatures;

typedef Eigen::Triplet<double> sptriple;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;

namespace OGT_NAMESPACE {
namespace embed {

/// Finds the index of the specified object pair.
Index pair(size_t a, size_t b, size_t nObj) {
	Index i = min(a, b), j = max(a, b);
	return (nObj * (nObj-1) / 2)
		- ((nObj - i) * (nObj - i - 1) / 2)
		+ (j - i - 1);
}

/// Computes a kernel matrix from a sparse feature matrix.
/// Computes the k1 kernel matrix from the paper cited for embedCmpWithTauForK()
MatrixXd embedCmpWithTauK1(const vector<CmpConstraint>& cons, size_t nObj) {
	vector<sptriple> triplets;
	triplets.reserve(cons.size());
	VectorXd norm = VectorXd::Zero(nObj);
	Index nPairs = nObj * (nObj - 1) / 2;
	for (const auto& con : cons) {
		assert(con.a < nObj && con.b < nObj && con.c < nObj);
		Index pr = pair(con.b, con.c, nObj);
		triplets.emplace_back(con.a, pr, (con.b < con.c) ? +1 : -1);
		norm(con.a)++;
	}
	SpMat phi(nObj, nPairs);
	phi.setFromTriplets(triplets.begin(), triplets.end());
	for (size_t obj = 0; obj < nObj; obj++) {
		if (norm(obj) > 0) {
			phi.row(obj) /= sqrt(norm(obj));
		}
	}
	return kernelForFeatures(phi);
}

/// Computes the k2 kernel matrix from the paper cited for embedCmpWithTauForK()
MatrixXd embedCmpWithTauK2(const vector<CmpConstraint>& cons, size_t nObj) {
	vector<sptriple> triplets;
	triplets.reserve(2 * cons.size());
	VectorXd norm = VectorXd::Zero(nObj);
	Index nPairs = nObj * (nObj - 1) / 2;
	for (const auto& con : cons) {
		assert(con.a < nObj && con.b < nObj && con.c < nObj);
		Index pr = pair(con.a, con.c, nObj);
		triplets.emplace_back(con.b, pr, +1);
		norm(con.b)++;

		pr = pair(con.a, con.b, nObj);
		if (pr >= nPairs) {
			std::cerr << "pair " << pr << " > " << nPairs << std::endl;
		}
		triplets.emplace_back(con.c, pr, -1);
		norm(con.c)++;
	}
	SpMat phi(nObj, nPairs);
	phi.setFromTriplets(triplets.begin(), triplets.end());
	for (size_t obj = 0; obj < nObj; obj++) {
		if (norm(obj) > 0) {
			phi.row(obj) /= sqrt(norm(obj));
		}
	}
	return kernelForFeatures(phi);
}

/// Compute a kernel matrix where K_ij is derived by how similarly points i and
/// j are ranked by the provided comparisons.
EmbedResult embedCmpWithTauForK(vector<CmpConstraint> cons, double k1,
	double k2) {
	size_t nObj = 0;
	for (const auto& con : cons) {
		nObj = max({nObj, con.a, con.b, con.c});
	}
	nObj++;
	EmbedResult result;
	result.K = MatrixXd::Zero(nObj, nObj);
	if (k1 > 0) {
		result.K += k1 * embedCmpWithTauK1(cons, nObj);
	}
	if (k2 > 0) {
		result.K += k2 * embedCmpWithTauK2(cons, nObj);
	}
	return result;
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
