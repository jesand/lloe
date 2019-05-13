/// @file  psd.cpp
/// @brief Implements geometric primitives.

#include <ogt/linalg/linalg.hpp>

using Eigen::EigenSolver;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::SelfAdjointEigenSolver;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using std::max;
using std::shared_ptr;
using std::vector;

namespace OGT_NAMESPACE {
namespace linalg {

/// Compute the eigenvalues and eigenvectors of a matrix.
EigResult eigendecomposition(const MatrixXd& matrix, bool assumeSelfAdjoint) {
	EigResult res;
	if (assumeSelfAdjoint || matrix == matrix.adjoint()) {
		auto solver = SelfAdjointEigenSolver<MatrixXd>(matrix);
		if (solver.info() != Eigen::Success) {
			throw LinAlgErr("Eigendecomposition failed with error code "
				+ std::to_string(solver.info()));
		}
		res.eig = solver.eigenvalues();
		res.vec = solver.eigenvectors();
	} else {
		auto solver = EigenSolver<MatrixXd>(matrix);
		if (solver.info() != Eigen::Success) {
			throw LinAlgErr("Eigendecomposition failed with error code "
				+ std::to_string(solver.info()));
		}
		res.eig = solver.eigenvalues();
		res.vec = solver.eigenvectors();
	}
	return res;
}

/// Ask whether a matrix is positive semidefinite.
bool isPSD(const MatrixXd& matrix, double eps) {
	if (matrix.rows() != matrix.cols()) {
		return false;
	}
	if (matrix != matrix.adjoint()) {
		return false;
	}
	auto eig = matrix.selfadjointView<Eigen::Lower>().eigenvalues().array();
	return ((eig + eps) >= 0).all();
}

/// Attempt to project a matrix onto the PSD cone.
MatrixXd projectOntoPSDCone(const MatrixXd& matrix, bool assumeSelfAdjoint) {

	// Find the eigenvalues and eigenvectors of the matrix
	auto eigc = eigendecomposition(matrix, assumeSelfAdjoint);

	// Find the real positive eigenvalues
	auto eig = eigc.eig.real();
	auto vec = eigc.vec.real();
	vector<Index> pos;
	size_t nPos = 0;
	for (Index ii = 0; ii < eig.size(); ii++) {
		if (eig(ii) > 0) {
			pos.push_back(ii);
			nPos++;
		} else if (eig(ii) > -1e-9) {
			nPos++;
		}
	}
	if (pos.empty()) {
		throw LinAlgErr("projectOntoPSDCone() failed: no positive eigenvalues");
	} else if (nPos == static_cast<size_t>(eig.size())) {
		// All eigenvalues are positive: matrix was already PSD
		return matrix;
	}

	// Project the matrix
	MatrixXd posEig = MatrixXd::Zero(pos.size(), pos.size());
	MatrixXd posVec(matrix.rows(), pos.size());
	for (Index ii = 0; ii < matrix.rows(); ii++) {
		if (ii < posEig.rows()) {
			posEig(ii, ii) = eig(pos[ii]);
		}
		for (Index jj = 0; jj < posVec.cols(); jj++) {
			posVec(ii,jj) = vec(ii,pos[jj]);
		}
	}
	MatrixXd psdm = posVec * posEig * posVec.transpose();

	// Sanity-check the result
	if (psdm.array().isNaN().any()) {
		throw LinAlgErr("projectOntoPSDCone() failed: Metric contains NaN values");
	}
	if (psdm.array().isInf().any()) {
		throw LinAlgErr("projectOntoPSDCone() failed: Metric contains Inf values");
	}
	return psdm;
}

/// Project a matrix onto the nearest matrix of the specified rank.
/// This works by setting the smallest eigenvalues to zero.
MatrixXd projectOntoLowRank(const MatrixXd& matrix, size_t rank) {
	if (matrix.rows() != matrix.cols()
		|| static_cast<size_t>(matrix.rows()) <= rank) {
		return matrix;
	}
	auto eig = eigendecomposition(matrix, true);
	VectorXd vals(eig.eig.real());
	MatrixXd vecs(eig.vec.real());
	vals.segment(rank, vals.size() - rank).setZero();
	return vecs * vals.asDiagonal() * vecs.transpose();
}

/// Project a matrix onto the nuclear norm ball.
/// This has the effect of reducing its rank, and works by reducing the smallest
/// singular values.
///
/// Implementation based on:
/// https://github.com/lalitkumarj/FORTE/blob/master/FORTE/algorithms/NuclearNormPGD.pyx
MatrixXd projectOntoNuclearNorm(const MatrixXd& matrix, double lambda) {
	auto eig = eigendecomposition(matrix, true);
	VectorXd vals(eig.eig.real());
	MatrixXd vecs(eig.vec.real());

	// Project the eigenvalues onto the simplex, unless it already is
	if (vals.sum() != lambda || (vals.array() < 0).any()) {

		// Find the cumulative sum of eigenvalues (in descending order)
		VectorXd cssv(vals.size());
		cssv(0) = vals(0);
		for (Index ii = 1; ii < vals.size(); ii++) {
			cssv(ii) = cssv(ii - 1) + vals(ii);
		}

		// Count > 0 components for the optimal solution
		Index rho = 0;
		for (; vals(rho) * (rho + 1) > cssv(rho) - lambda; rho++) {}
		rho--; // The above loop goes one step too far

		// Compute the Lagrange multiplier associated to the simplex constraint
		double theta = (cssv(rho) - lambda) / (rho + 1.0);

		// Compute the projection by thresholding vals using theta
		vals = (vals.array() - theta).max(0);
	}

	return vecs * vals.asDiagonal() * vecs.transpose();
}

/// Project a matrix onto the unit sphere.
/// All vectors are scaled to unit length.
MatrixXd projectOntoUnitSphere(const MatrixXd& matrix) {
	MatrixXd result = matrix;
	for (Index rr = 0; rr < result.rows(); rr++) {
		result.row(rr) /= result.row(rr).norm();
	}
	return result;
}

/// Implements PositionMatch.
class PositionMatchImpl : public PositionMatch {
public:
	virtual ~PositionMatchImpl() = default;

	/// Constructor.
	PositionMatchImpl(MatrixXd rotation, VectorXd translation, double scaling)
		: rotation(rotation), translation(translation), scaling(scaling) {}

	/// Transforms a matrix in-place into the target space.
	void transform(MatrixXd& matrix) override {
		matrix = scaling * matrix * rotation
			+ VectorXd::Ones(matrix.rows()) * translation.transpose();
	}

	/// Return the rotation matrix.
	MatrixXd rotationMatrix() const override {
		return rotation;
	}

	/// Return the translation vector.
	VectorXd translationVector() const override {
		return translation;
	}

	/// Return the scaling factor.
	double scalingFactor() const override {
		return scaling;
	}

private:

	/// The rotation matrix.
	MatrixXd rotation;

	/// The translation vector.
	VectorXd translation;

	/// The scaling factor.
	double scaling;
};

/// Finds a Procrustes transformation which minimizes the sum of squared
/// distances between corresponding rows of target and testee.
///
/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
///     and applications. Springer.
shared_ptr<PositionMatch> PositionMatch::Create(const MatrixXd& target,
	const MatrixXd& testee) {
	Index nObj = target.rows();

	// J is a centering matrix
	MatrixXd J = MatrixXd::Identity(nObj, nObj)
		- MatrixXd::Constant(nObj, nObj, 1.0 / nObj);

	// Compute the SVD of X' J Y, to derive our transformation.
	MatrixXd C = target.transpose() * J * testee;
	auto svd = C.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	MatrixXd rotation = svd.matrixV() * svd.matrixU().transpose();
	double scaling = (target.transpose() * J * testee * rotation).trace()
		/ (testee.transpose() * J * testee).trace();
	VectorXd translation = (1.0 / nObj)
		* (target - scaling * testee * rotation).transpose()
		* VectorXd::Ones(nObj);
	return std::make_shared<PositionMatchImpl>(rotation, translation, scaling);
}

/// Find the distance scaling which minimizes the difference between two
/// distance matrices, in a least-squares sense.
double distScalingFactor(const MatrixXd& dhat, const MatrixXd& dtrue) {
	assert(dhat.rows() == dhat.cols());
	assert(dhat.rows() == dtrue.rows() && dhat.cols() == dtrue.cols());
	size_t nObj = dtrue.rows();
	size_t numel = nObj * (nObj - 1) / 2;
	VectorXd vhat(numel), vtrue(numel);
	Index next = 0;
	for (size_t i = 1; i < nObj; i++) {
		for (size_t j = 0; j < i; j++) {
			vhat(next) = dhat(i,j);
			vtrue(next) = dtrue(i,j);
			next++;
		}
	}
	assert(next == static_cast<Index>(numel));
	return (vhat.transpose() * vtrue).value()
		/ (vhat.transpose() * vhat).value();
}

template<>
SparseMatrix<double, RowMajor> selectRows(const SparseMatrix<double, RowMajor>& X,
	vector<size_t> rows) {
	typedef Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(X.nonZeros());
	if (X.IsRowMajor) {
		for (size_t ii = 0; ii < rows.size(); ++ii) {
			Index row = rows[ii];
			for (SparseMatrix<double, RowMajor>::InnerIterator it(X, row); it; ++it) {
				tripletList.emplace_back(ii, it.col(), it.value());
			}
		}
	} else {
		for (size_t ii = 0; ii < rows.size(); ++ii) {
			Index row = rows[ii];
			for (Index col = 0; col < X.cols(); col++) {
				double val = X.coeff(row, col);
				if (val != 0) {
					tripletList.emplace_back(row, col, val);
				}
			}
		}
	}
	SparseMatrix<double, RowMajor> result(rows.size(), X.cols());
	result.setFromTriplets(tripletList.begin(), tripletList.end());
	return result;
}

/// Center a matrix
MatrixXd centerMatrix(const MatrixXd& X) {
	return X.rowwise() - X.colwise().mean();
}

/// Center a position matrix about the origin, and scale it to unit diameter
MatrixXd normalizePos(const MatrixXd& X) {
	MatrixXd out = centerMatrix(X);
	double scale = 0;
	for (Index dim = 0; dim < X.cols(); dim++) {
		scale = max(scale, X.col(dim).maxCoeff() - X.col(dim).minCoeff());
	}
	return scale ? out / scale : out;
}

/// Transform a vector by isotonic regression so its values are changed by the
/// smallest amount possible to appear in the specified order.
VectorXd isotonicRegression(const VectorXd& V, const VectorXd& weights,
	const vector<Index>& order) {
	VectorXd tmpV(V.size());
	VectorXd tmpW(weights.size());
	tmpV[0] = V[order[0]];
	tmpW[0] = weights[order[0]];
	Index jj = 0;
	vector<Index> outOrder(order.size() + 1, 0);
	outOrder[0] = -1;
	outOrder[1] = 0;
	for (Index ii = 1; ii < V.size(); ii++) {
		jj++;
		tmpV[jj] = V[order[ii]];
		tmpW[jj] = weights[order[ii]];
		for (; jj > 0 && tmpV[jj] < tmpV[jj - 1]; jj--) {
			tmpV[jj - 1] = (tmpW[jj] * tmpV[jj] + tmpW[jj - 1] * tmpV[jj - 1])
				/ (tmpW[jj] + tmpW[jj - 1]);
			tmpW[jj - 1] += tmpW[jj];
		}
		outOrder[jj + 1] = ii;
	}
	VectorXd out(V.size());
	for (Index kk = 1; kk <= jj + 1; kk++) {
		for (Index ll = outOrder[kk - 1] + 1; ll <= outOrder[kk]; ll++) {
			out[order[ll]] = tmpV[kk - 1];
		}
	}
	return out;	
}

/// Get pairwise squared Euclidean distances from a dissimilarity kernel.
MatrixXd sqDistsFromKernel(const MatrixXd& K) {
	assert(K.rows() == K.cols());
	MatrixXd D(K.rows(), K.cols());
	for (Index ii = 0; ii < K.rows(); ii++) {
		D(ii,ii) = 0;
		for (Index jj = 0; jj < ii; jj++) {
			D(ii,jj) = sqDistFromKernel(K, ii, jj);
			D(jj,ii) = D(ii,jj);
		}
	}
	return D;
}

/// Get an embedding from a distance kernel via SVD
MatrixXd embeddingFromKernelSVD(const MatrixXd& K, size_t nDim) {
	assert(K.rows() == K.cols());
	auto svd = K.bdcSvd(Eigen::ComputeFullU);
	MatrixXd X = svd.matrixU().leftCols(nDim);
	VectorXd S = svd.singularValues().segment(0, nDim).cwiseSqrt();
	for (Index row = 0; row < X.rows(); row++) {
		VectorXd newrow = X.row(row).cwiseProduct(S.transpose());
		X.row(row) = newrow;
	}
	return X;
}

/// Find the point nearest the intersection of k spheres.
VectorXd sphereIntersection(const MatrixXd& centers, const VectorXd& radii,
	double& quality) {
	assert(centers.cols() == radii.size());
	const size_t numDim = centers.rows();
	const size_t numCent = centers.cols();
	if (numCent == 1) {
		quality = 0;
		VectorXd v(1); v << radii(0) + centers(0, 0);
		return v;
	}
	VectorXd x0 = centers.col(0);
	MatrixXd A = (centers.colwise() - x0).rightCols(numCent - 1);
	MatrixXd A2 = A * (A.transpose() * A).inverse();
	MatrixXd P = A2 * A.transpose();

	VectorXd norm = centers.colwise().squaredNorm();
	VectorXd rsq = radii.cwiseProduct(radii);
	VectorXd b = ((norm - rsq).array() + rsq(0) - norm(0)).tail(numCent - 1) / 2;

	VectorXd v = (MatrixXd::Identity(numCent - 1, numCent - 1) - P)
		* x0 + A2 * b;
	quality = rsq(0) - (x0 - v).squaredNorm();

	if (quality > 1e-9) {
		v.conservativeResize(numDim + 1);
		v(numDim) = sqrt(quality);
	}
	return v;
}


} // end namespace linalg
} // end namespace OGT_NAMESPACE
