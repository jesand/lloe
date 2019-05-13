/// @file  dist.cpp
/// @brief Implements distance embedding methods.

#include <ogt/embed/dist.hpp>
#include <ogt/linalg/linalg.hpp>
#include <algorithm>
#include <array>
#include <numeric>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using OGT_NAMESPACE::linalg::eigendecomposition;
using std::array;
using std::min;
using std::vector;

namespace OGT_NAMESPACE {
namespace embed {

/// A helper for the embedDist overrides.
MatrixXd embedDistHelper(const MatrixXd& dist, size_t nDim, double lambda) {
	const Index nObj = dist.rows();
	MatrixXd J = MatrixXd::Identity(nObj, nObj)
		- MatrixXd::Constant(nObj, nObj, 1.0 / nObj);
	auto eig = eigendecomposition(-0.5 * J * dist.cwiseProduct(dist) * J);
	VectorXd eigVal = eig.eig.real();
	if (nDim == 0) {
		for (nDim = 1; eigVal(nDim) >= lambda; nDim++) {}
	} else {
		size_t maxDim;
		for (maxDim = 1; maxDim < static_cast<size_t>(nObj)
			&& eigVal(maxDim) > 0; maxDim++) {}
		nDim = min(nDim, maxDim);
	}
	eigVal = eigVal.cwiseSqrt();
	return eig.vec.real().leftCols(nDim) * eigVal.segment(0, nDim).asDiagonal();
}

/// Produce an embedding of the specified dimensionality from the given
/// Euclidean distance matrix.
MatrixXd embedDistWithDims(const MatrixXd& dist, size_t nDim) {
	return embedDistHelper(dist, nDim, 0);
}

/// Produce an embedding from the given Euclidean distance matrix.
MatrixXd embedDist(const MatrixXd& dist, double lambda) {
	return embedDistHelper(dist, 0, lambda);
}

/// Produce an embedding into the target dimensionality, attempting to preserve
/// the total ordering of pairwise distances.
/// An ordinal matrix is produced from the given distance matrix, and an
/// eigendecomposition is used to embed the points.
///
/// Implements method from:
/// [1] Dattorro, Jon, "Convex Optimization and Euclidean Distance Geometry,"
///     Meboo Publishing, 2016.
MatrixXd embedDistWithOrdEig(const MatrixXd& dist, size_t nDim) {

	// Build the order matrix
	const Index nObj = dist.cols();
	const double nPairs = (nObj - 1) * nObj / 2;
	auto pos = [=](Index idx) -> array<Index,2> {
		// equation via: http://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
		array<Index,2> ij;
		ij[0] = nObj - 2 - floor(
			sqrt( -8 * idx + 4 * nObj * (nObj - 1) - 7) / 2.0 - 0.5);
		ij[1] = idx + ij[0] + 1 - nObj * (nObj - 1) / 2
			+ (nObj - ij[0]) * ((nObj - ij[0]) - 1) / 2;
		return ij;
	};
	vector<Index> order(nObj * (nObj - 1) / 2);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](auto a, auto b) -> bool {
		auto posA = pos(a);
		auto posB = pos(b);
		return dist(posA[0], posA[1]) < dist(posB[0], posB[1]);
	});
	MatrixXd ordMatrix = MatrixXd::Zero(nObj, nObj);
	for (size_t i = 0; i < order.size(); i++) {
		auto ij = pos(order[i]);
		ordMatrix(ij[0], ij[1]) = (i + 1) / nPairs;
		ordMatrix(ij[1], ij[0]) = ordMatrix(ij[0], ij[1]);
	}

	// Build the VOV matrix
	MatrixXd Vn(nObj, nObj-1);
	Vn.row(0) = VectorXd::Constant(nObj-1, -1);
	Vn.bottomRows(nObj-1) = MatrixXd::Identity(nObj-1, nObj-1);
	MatrixXd VOV = (-Vn.transpose() * ordMatrix * Vn) / 2;

	// Find an eigendecomposition of VOV
	auto eig = eigendecomposition(VOV);

	// Calculate the embedding
	MatrixXd XT(nDim,nObj);
	XT.col(0) = VectorXd::Zero(nDim);
	nDim = std::min<size_t>(nDim, eig.eig.real().size());
	VectorXd eigs = eig.eig.real().segment(0,nDim);
	MatrixXd L = eigs.cwiseSqrt().asDiagonal();
	MatrixXd R = eig.vec.real().leftCols(nDim).transpose();
	XT.rightCols(nObj-1) = L * R;
	return XT.transpose();
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
