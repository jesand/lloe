/// @file  cmdet.cpp
/// @brief Implements Cayley-Menger coordinate transforms.

#include <ogt/embed/cmdet.hpp>
#include <cassert>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::min;
using std::shared_ptr;

namespace OGT_NAMESPACE {
namespace embed {

/// Concrete implementation of CMReference.
class CMReferenceImpl : public CMReference {
public:

	/// Constructor.
	CMReferenceImpl(MatrixXd refSqDists) {
		assert(refSqDists.rows() == refSqDists.cols());

		// Build the matrix N of determinants, which builds the reference
		Index nDim = refSqDists.cols() - 1;
		MatrixXd MR = MatrixXd::Ones(nDim + 2, nDim + 2);
		MR(0,0) = 0;
		MR.block(1, 1, nDim + 1, nDim + 1) = refSqDists;

		MatrixXd N = MatrixXd::Zero(nDim + 2, nDim + 2);
		for (Index kk = 0; kk < N.rows(); kk++) {
			for (Index ll = 0; ll < N.cols(); ll++) {
				MatrixXd M2(nDim + 1, nDim + 1);
				M2.topLeftCorner(ll, kk) = MR.topLeftCorner(ll, kk);
				M2.topRightCorner(ll, nDim + 1 - kk)
					= MR.topRightCorner(ll, nDim + 1 - kk);
				M2.bottomLeftCorner(nDim + 1 - ll, kk)
					= MR.bottomLeftCorner(nDim + 1 - ll, kk);
				M2.bottomRightCorner(nDim + 1 - ll, nDim + 1 - kk)
					= MR.bottomRightCorner(nDim + 1 - ll, nDim + 1 - kk);
				N(kk,ll) = M2.determinant();
			}
		}

		// Now construct the reference elements
		A = MatrixXd::Zero(nDim + 1, nDim + 1);
		b = VectorXd::Zero(nDim + 1);
		for (Index kk = 0; kk < A.rows(); kk++) {
			for (Index ll = 0; ll < A.cols(); ll++) {
		        A(kk,ll) = pow(-1, kk + ll + 1) * N(kk + 1, ll + 1);
		    }
		    b(kk) = pow(-1, kk) * N(0, kk + 1);
		}
		c = -N(0, 0);
		cmr = MR.determinant();
	}

	/// Virtual destructor.
	virtual ~CMReferenceImpl() = default;

	/// Find the squared Euclidean distance between two points with the given
	/// squared Euclidean distances to the reference vertices.
	double sqDist(VectorXd v1, VectorXd v2) override {
		return -(v1.transpose() * A * v2 + b.dot(v1 + v2) + c) / cmr;
	}

private:
	MatrixXd A;
	VectorXd b;
	double c, cmr;
};


/// Construct a concrete instance, given all pairwise distances between the
/// reference vertices.
shared_ptr<CMReference> CMReference::Create(MatrixXd refDists) {
	return std::make_shared<CMReferenceImpl>(refDists);
}

/// Produce an embedding using Euclidean distances to a subset of the set.
/// The matrix dist must be a n x (d+1) Euclidean distance matrix, where each
/// column represents the distance to some reference vertex.
/// The produced embedding will be in R^d.
///
/// @see CMReference
MatrixXd embedDistWithCMReference(const MatrixXd& dist, size_t maxDim) {
	MatrixXd sqDist = dist.cwiseProduct(dist);

	// Choose the reference vertex with the max min distance to the others as
	// the origin.
	VectorXd minDist = VectorXd::Constant(sqDist.cols(), INFINITY);
	for (auto ii = 0; ii < sqDist.cols(); ii++) {
		for (auto jj = 0; jj < sqDist.cols(); jj++) {
			minDist(ii) = min(minDist(ii), sqDist(ii,jj));
		}
	}
	Index origin = 0;
	double maxMinDist = minDist(0);
	for (auto ii = 1; ii < sqDist.cols(); ii++) {
		if (minDist(ii) > maxMinDist) {
			maxMinDist = minDist(ii);
			origin = ii;
		}
	}
	if (origin > 0) {
		sqDist.col(0).swap(sqDist.col(origin));
		sqDist.row(0).swap(sqDist.row(origin));
	}

	// Now embed, choosing reference vertices whose remaining orthogonal
	// distance to the origin is maximized.
	MatrixXd X(sqDist.rows(), sqDist.cols() - 1);
	Index nDim = min<Index>(X.cols(), maxDim);
	for (Index dim = 0; dim < nDim; dim++) {

		// Choose the next reference vertex with the largest remaining distance
		// to the origin.
		Index maxRef = dim + 1;
		double maxDist = sqDist(maxRef, 0);
		for (Index ref = maxRef + 1; ref < sqDist.cols(); ref++) {
			if (sqDist(ref, 0) > maxDist) {
				maxRef = ref;
				maxDist = sqDist(ref, 0);
			}
		}
		if (maxDist <= 0) {
			nDim = dim;
			break;
		} else if (maxRef > dim + 1) {
			sqDist.col(dim + 1).swap(sqDist.col(maxRef));
			sqDist.row(dim + 1).swap(sqDist.row(maxRef));
		}

		// Project each object onto this axis
		Index ref = dim + 1;
		X(0, dim) = 0;
		X(ref, dim) = sqrt(sqDist(ref, 0));
		for (Index obj = 1; obj < X.rows(); obj++) {
			if (obj != ref) {
	            X(obj, dim) = (sqDist(obj, 0) + sqDist(ref, 0)
	            	- sqDist(obj, ref)) / (2 * X(ref, dim));
			}
		}

		// Keep just the orthogonal components of each distance
		sqDist.col(0) = sqDist.col(0) - X.col(dim).cwiseProduct(X.col(dim));
		sqDist.col(ref) = sqDist.col(0);
		for (Index obj = 0; obj < X.rows(); obj++) {
			for (Index col = 1; col < sqDist.cols(); col++) {
				if (col == ref) continue;
				if (obj == 0 || obj == ref) {
					sqDist(obj, col) = sqDist(col, 0);
				} else {
					sqDist(obj, col) -= pow(X(col, dim) - X(obj, dim), 2);
				}
			}
		}
	}
	return nDim == X.cols() ? X : X.leftCols(nDim);
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
