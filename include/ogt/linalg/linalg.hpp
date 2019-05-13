/// @file  linalg.hpp
/// @brief Linear algebra routines.

#pragma once
#ifndef OGT_LINALG_LINALG_HPP
#define OGT_LINALG_LINALG_HPP

#include <ogt/config.hpp>
#include <Eigen/Eigen>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace OGT_NAMESPACE {
namespace linalg {

/// An exception to throw in case of mathematical error in some linear algebra
/// routine.
struct LinAlgErr : public std::runtime_error {
	virtual ~LinAlgErr() = default;

	/// Build an LinAlgErr with the specified message.
	LinAlgErr(std::string message)
		: std::runtime_error("ogt::LinAlgErr: " + message) {
	}
};

/// The eigenvalues and eigenvectors of a matrix
struct EigResult {
	virtual ~EigResult() = default;

	/// The eigenvalues
	Eigen::ArrayXcd eig;

	/// The eigenvectors
	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType vec;
};

/// Iterates over nonzero entries in a dense vector.
template<typename T>
struct DenseNZIterator {
	DenseNZIterator(const T& vector) : _vec(vector), _idx(-1) { ++(*this); }
	virtual ~DenseNZIterator() = default;
	void operator++() {
		++_idx;
		while (_idx < _vec.size() && _vec[_idx] == 0) {
			++_idx;
		}
	}
	operator bool() const {
		return _idx < _vec.size();
	}
	size_t index() const {
		return static_cast<size_t>(_idx);
	}

	const T& _vec;
	int _idx;
};

/// Compute the eigenvalues and eigenvectors of a matrix.
/// Throws LinAlgErr on failure.
EigResult eigendecomposition(const Eigen::MatrixXd& matrix,
	bool assumeSelfAdjoint = false);

/// Ask whether a matrix is positive semidefinite.
/// This tests whether the eigenvalues are non-negative, to within eps
/// precision.
/// Any PSD matrix is a valid n x n distance matrix for some Euclidean space
/// R^d, with d <= n - 1.
bool isPSD(const Eigen::MatrixXd& matrix, double eps = 1e-12);

/// Attempt to project a matrix onto the PSD cone.
/// Throws LinAlgErr in case of failure.
/// This will produce a PSD matrix which is as close as possible to the input
/// matrix. This is often used with methods which compute matrices to satisfy
/// some loss minimization objective without constraining those matrices to be
/// distance matrices. The output from such a method can be projected onto the
/// PSD cone to obtain the PSD solution which is as close as possible to the
/// solution with minimal loss. This is sometimes faster than constraining the
/// optimization problem to stay within the PSD cone.
Eigen::MatrixXd projectOntoPSDCone(const Eigen::MatrixXd& matrix,
	bool assumeSelfAdjoint);

/// Project a matrix onto the nearest matrix of the specified rank.
/// This works by setting the smallest eigenvalues to zero.
Eigen::MatrixXd projectOntoLowRank(const Eigen::MatrixXd& matrix, size_t rank);

/// Project a matrix onto the nuclear norm ball.
/// This has the effect of reducing its rank.
///
/// [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions.
///     John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
///     International Conference on Machine Learning (ICML 2008)
///     http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
Eigen::MatrixXd projectOntoNuclearNorm(const Eigen::MatrixXd& matrix,
	double lambda);

/// Project a matrix onto the unit sphere.
/// All vectors are scaled to unit length.
Eigen::MatrixXd projectOntoUnitSphere(const Eigen::MatrixXd& matrix);

/// A position match is a rotation/reflection, translation, and scaling of a
/// position matrix so that its points are as close as possible to the positions
/// of the corresponding rows in some target matrix.
/// This is also known as a Procrustes transformation.
struct PositionMatch {
	virtual ~PositionMatch() = default;

	/// Finds a Procrustes transformation which minimizes the sum of squared
	/// distances between corresponding rows of target and testee.
	///
	/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
	///     and applications. Springer.
	static std::shared_ptr<PositionMatch> Create(const Eigen::MatrixXd& target,
		const Eigen::MatrixXd& testee);

	/// Transforms a matrix in-place into the target space.
	virtual void transform(Eigen::MatrixXd& matrix) = 0;

	/// Return the rotation matrix.
	virtual Eigen::MatrixXd rotationMatrix() const = 0;

	/// Return the translation vector.
	virtual Eigen::VectorXd translationVector() const = 0;

	/// Return the scaling factor.
	virtual double scalingFactor() const = 0;
};

/// Find the distance scaling which minimizes the difference between two
/// distance matrices, in a least-squares sense.
double distScalingFactor(const Eigen::MatrixXd& dhat,
	const Eigen::MatrixXd& dtrue);

/// Select the specified rows of a matrix.
template<typename mat_type>
mat_type selectRows(const mat_type& X, std::vector<size_t> rows) {
	mat_type result(rows.size(), X.cols());
	for (size_t ii = 0; ii < rows.size(); ii++) {
		result.row(ii) = X.row(rows[ii]);
	}
	return result;
}
template<>
Eigen::SparseMatrix<double, Eigen::RowMajor> selectRows(
	const Eigen::SparseMatrix<double, Eigen::RowMajor>& X,
	std::vector<size_t> rows);

/// Convert a position matrix into a Euclidean distance matrix.
template<typename mat_type>
Eigen::MatrixXd posToDist(const mat_type& X) {
	Eigen::MatrixXd B = X * X.transpose();
	Eigen::VectorXd c = B.diagonal();
	Eigen::VectorXd one = Eigen::VectorXd::Ones(X.rows());
	return (c * one.transpose() + one * c.transpose() - 2 * B).cwiseSqrt();
}

/// Convert a position matrix into a Cosine distance matrix.
template<typename mat_type>
Eigen::MatrixXd posToCosineDist(const mat_type& X) {
	Eigen::MatrixXd dists(X.rows(), X.rows());
	Eigen::VectorXd norm(X.rows());
	for (Eigen::Index ii = 0; ii < X.rows(); ii++) {
		norm(ii) = X.row(ii).norm();
	}
	for (Eigen::Index ii = 0; ii < X.rows(); ii++) {
		auto a = X.row(ii);
		for (Eigen::Index jj = 0; jj < ii; jj++) {
			auto b = X.row(jj);
			dists(ii,jj) = 1.0 - (a.dot(b) / (norm(ii) + norm(jj)));
			dists(jj,ii) = dists(ii,jj);
		}
	}
	return dists;
}

/// Convert a position matrix into a Jaccard distance matrix.
template<typename M, typename V, typename IT>
Eigen::MatrixXd posToJaccardDist(const M& X) {
	Eigen::MatrixXd dists(X.rows(), X.rows());
	for (Eigen::Index ii = 0; ii < X.rows(); ii++) {
		V a = X.row(ii);
		for (Eigen::Index jj = 0; jj < ii; jj++) {
			V b = X.row(jj);

			IT ita(a), itb(b);
			double inBoth = 0, total = 0;
			while (ita && itb) {
				total++;
				if (ita.index() < itb.index()) {
					++ita;
				} else if (ita.index() > itb.index()) {
					++itb;
				} else {
					inBoth++;
					++ita;
					++itb;
				}
			}
			for (; ita; ++ita) {
				total++;
			}
			for (; itb; ++itb) {
				total++;
			}

			dists(ii,jj) = total ? (total - inBoth) / total : INFINITY;
			dists(jj,ii) = dists(ii,jj);
		}
	}
	return dists;
}

/// Center a matrix
Eigen::MatrixXd centerMatrix(const Eigen::MatrixXd& X);

/// Center a position matrix about the origin, and scale it to unit diameter
Eigen::MatrixXd normalizePos(const Eigen::MatrixXd& X);

/// Transform a vector by isotonic regression so its values are changed by the
/// smallest amount possible to appear in the specified order.
///
/// Implements the PAVA algorithm with uniform weights, as described here:
/// http://stat.wikia.com/wiki/Isotonic_regression
Eigen::VectorXd isotonicRegression(const Eigen::VectorXd& V,
	const Eigen::VectorXd& weights, const std::vector<Eigen::Index>& order);

/// Calculate the Euclidean distance between two points.
inline double dist(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
	assert(a < X.rows());
	assert(b < X.rows());
	return (X.row(a) - X.row(b)).norm();
}

/// Calculate the squared Euclidean distance between two points.
inline double sqDist(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
	assert(a < X.rows());
	assert(b < X.rows());
	return (X.row(a) - X.row(b)).squaredNorm();
}

/// Get pairwise squared Euclidean distances from a dissimilarity kernel.
Eigen::MatrixXd sqDistsFromKernel(const Eigen::MatrixXd& K);

/// Calculate the squared Euclidean distance between two points from a kernel.
inline double sqDistFromKernel(const Eigen::MatrixXd K, Eigen::Index a,
	Eigen::Index b) {
	assert(K.rows() == K.cols());
	assert(a < K.rows());
	assert(b < K.rows());
	return K(a,a) + K(b,b) - 2 * K(a,b);
}

/// Get an embedding from a dissimilarity kernel via SVD
Eigen::MatrixXd embeddingFromKernelSVD(const Eigen::MatrixXd& K, size_t nDim);

/// Compute a kernel matrix from a given position/feature matrix.
template<typename mat_type>
Eigen::MatrixXd kernelForFeatures(const mat_type& mat) {
	Eigen::Index nObj = mat.rows();
	Eigen::MatrixXd kernel = Eigen::MatrixXd::Zero(nObj, nObj);
	for (Eigen::Index ii = 0; ii < nObj; ii++) {
		for (Eigen::Index jj = 0; jj <= ii; jj++) {
			kernel(ii,jj) = mat.row(ii).dot(mat.row(jj));
			if (ii != jj) {
				kernel(jj,ii) = kernel(ii,jj);
			}
		}
	}
	return kernel;
}

/// Find the intersection of spheres with k-dimensional center coordinates.
/// The sphere centers are given as the columns of centers.
/// The quality variable is negative if the spheres do not intersect,
/// zero if they intersect in R^k, and positive if they intersect in R^(k+1).
/// Citation: Thm. 3.3 from
///     H.X. Huang, Z.-A. Liang, and P. M. Pardalos,
///     "Some Properties for the Euclidean Distance Matrix and Positive
///     Semidefinite Matrix Completion Problems,"
///     J Glob Optim, vol. 25, no. 1, pp. 3–21, 2003.
Eigen::VectorXd sphereIntersection(const Eigen::MatrixXd& centers,
	const Eigen::VectorXd& radii, double& quality);


} // end namespace linalg
} // end namespace OGT_NAMESPACE
#endif /* OGT_LINALG_LINALG_HPP */
