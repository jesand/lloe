/// @file  embed_cmp_with_pgd.cpp
/// @brief Implements the projected gradient descent methods of Jain et al.,
///        NIPS 2016.

#include <ogt/embed/cmp.hpp>
#include <ogt/embed/dlib_opt.hpp>
#include <ogt/linalg/linalg.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using OGT_NAMESPACE::linalg::eigendecomposition;
using OGT_NAMESPACE::linalg::projectOntoLowRank;
using OGT_NAMESPACE::linalg::projectOntoNuclearNorm;
using OGT_NAMESPACE::linalg::sqDist;
using std::vector;

namespace OGT_NAMESPACE {
namespace embed {

/// The data container for the PGD objective information
struct PGDBase {

	/// Constructor
	PGDBase(const vector<CmpConstraint>& cons)
		: cons(cons) {}

	const vector<CmpConstraint>& cons;
};

/// The PGD risk objective.
struct PGDObjectiveForX : PGDBase {

	/// Constructor
	PGDObjectiveForX(const vector<CmpConstraint>& cons) : PGDBase(cons) {}

	/// Calculate the objective value
	double operator()(const MatrixXd& X) const {
		
		// y is -1 when the triple dist(a,b) < dist(a,c) is observed, and is
		// +1 otherwise. We arrange the constraints so that y = -1 always.
		const double y = -1; 

		double risk = 0;
		for (const auto& con : cons) {
			// lt = dist(a,b) - dist(a,c)
			double lt = sqDist(X, con.a, con.b) - sqDist(X, con.a, con.c);
			risk += log(1 + exp(-y * lt));
		}
		risk /= cons.size();
		return risk;
	}
};

/// The gradient of the PGD risk objective.
struct PGDGradientForX : PGDBase {

	/// Constructor
	PGDGradientForX(const vector<CmpConstraint>& cons) : PGDBase(cons) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) const {

		// y is -1 when the triple dist(a,b) < dist(a,c) is observed, and is
		// +1 otherwise. We arrange the constraints so that y = -1 always.
		const double y = -1; 
		for (const auto& con : cons) {
			// lt = dist(a,b) - dist(a,c)
			double lt = sqDist(X, con.a, con.b) - sqDist(X, con.a, con.c);
			double p = 1.0 / (1.0 + exp(y * lt));

			VectorXd xa = X.row(con.a);
			VectorXd xb = X.row(con.b);
			VectorXd xc = X.row(con.c);
			grad.row(con.a) -= 2.0 * ((xa - xb) - (xa - xc)) * y * p;
			grad.row(con.b) += 2.0 * (xa - xb) * y * p;
			grad.row(con.c) -= 2.0 * (xa - xc) * y * p;
		}
		grad /= cons.size();
	}
};

/// The PGD risk objective.
struct PGDObjectiveForK : PGDBase {

	/// Constructor
	PGDObjectiveForK(const vector<CmpConstraint>& cons) : PGDBase(cons) {}

	/// Calculate the objective value
	double operator()(const MatrixXd& K) const {
		
		// y is -1 when the triple dist(a,b) < dist(a,c) is observed, and is
		// +1 otherwise. We arrange the constraints so that y = -1 always.
		const double y = -1; 

		double risk = 0;
		for (const auto& con : cons) {
			// lt = dist(a,b) - dist(a,c)
			double lt = K(con.b,con.b) - 2 * K(con.a,con.b)
				- K(con.c,con.c) + 2 * K(con.a,con.c);
			risk += log(1 + exp(-y * lt));
		}
		risk /= cons.size();
		return risk;
	}
};

/// The gradient of the PGD risk objective.
struct PGDGradientForK : PGDBase {

	/// Constructor
	PGDGradientForK(const vector<CmpConstraint>& cons) : PGDBase(cons) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& K, MatrixXd& grad) const {

		// y is -1 when the triple dist(a,b) < dist(a,c) is observed, and is
		// +1 otherwise. We arrange the constraints so that y = -1 always.
		const double y = -1; 
		for (const auto& con : cons) {
			// lt = dist(a,b) - dist(a,c)
			double lt = K(con.b,con.b) - 2 * K(con.a,con.b)
				- K(con.c,con.c) + 2 * K(con.a,con.c);
			double p = 1.0 / (1.0 + exp(y * lt));

			grad(con.b,con.b) -= y * p;
			grad(con.c,con.c) += y * p;
			grad(con.a,con.b) += 2 * y * p;
			grad(con.b,con.a) += 2 * y * p;
			grad(con.a,con.c) -= 2 * y * p;
			grad(con.c,con.a) -= 2 * y * p;
		}
		grad /= cons.size();
	}
};

/// The data container for the PGD objective information
struct DebiasBase {

	/// Constructor
	DebiasBase(const MatrixXd& vecs)
		: vecs(vecs) {}

	/// Get the kernel, given the specified eigenvalues
	MatrixXd calcK(const MatrixXd& S) const {
		return vecs * S.asDiagonal() * vecs.transpose();
	}

	const MatrixXd& vecs;
};

/// The PGD risk objective using fixed basis vectors.
struct DebiasObjective : DebiasBase {

	/// Constructor
	DebiasObjective(const vector<CmpConstraint>& cons,
		const MatrixXd& vecs) : DebiasBase(vecs), obj(cons) {}

	/// Calculate the objective value
	double operator()(const MatrixXd& S) const {
		return obj(calcK(S));
	}

	PGDObjectiveForK obj;
};

/// The gradient of the PGD risk objective using fixed basis vectors.
struct DebiasGradient : DebiasBase {

	/// Constructor
	DebiasGradient(const vector<CmpConstraint>& cons, const MatrixXd& vecs)
		: DebiasBase(vecs), cons(cons) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& S, MatrixXd& grad) const {

		// y is -1 when the triple dist(a,b) < dist(a,c) is observed, and is
		// +1 otherwise. We arrange the constraints so that y = -1 always.
		const double y = -1; 
		auto K = calcK(S);
		for (const auto& con : cons) {
			// lt = dist(a,b) - dist(a,c)
			double lt = K(con.b,con.b) - 2 * K(con.a,con.b)
				- K(con.c,con.c) + 2 * K(con.a,con.c);
			double p = 1.0 / (1.0 + exp(y * lt));

			VectorXd xa = vecs.row(con.a);
			VectorXd xb = vecs.row(con.b);
			VectorXd xc = vecs.row(con.c);
			grad -= (-2 * xa.cwiseProduct(xb) + xb.cwiseProduct(xb)
				+ 2 * xa.cwiseProduct(xc) - xc.cwiseProduct(xc)) * y * p;
		}
		grad /= cons.size();
	}

	const vector<CmpConstraint>& cons;
};

/// Embeds a dataset using Rank-d Projected Gradient Descent (PGD), adapted
/// to work directly on a n x d matrix X rather than to project after each step.
EmbedResult embedCmpWithPGDForX(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, EmbedConfig config) {
	auto result = optimizeMatrix(X0,
		PGDObjectiveForX(cons),
		PGDGradientForX(cons),
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Rank-d Projected Gradient Descent (PGD).
EmbedResult embedCmpWithPGDForK(vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, EmbedConfig config) {
	auto result = optimizeMatrixWithAdjustment(K0,
		PGDObjectiveForK(cons),
		PGDGradientForK(cons),
		[=](auto K) { return projectOntoLowRank(K, config.nDim); },
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.K = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Nuclear Norm Projected Gradient Descent.
EmbedResult embedCmpWithNNPGD(vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, double lambda, EmbedConfig config) {
	auto result = optimizeMatrixWithAdjustment(K0,
		PGDObjectiveForK(cons),
		PGDGradientForK(cons),
		[=](auto K) { return projectOntoNuclearNorm(K, lambda); },
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.K = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Nuclear Norm Projected Gradient Descent Debiased.
EmbedResult embedCmpWithNNPGDDebiased(vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, double lambda, EmbedConfig config) {

	// Find the (biased) optimal nuclear norm solution
	auto biasedResult = embedCmpWithNNPGD(cons, K0, lambda, config);

	// Debias the result with a second optimization
	auto eig = eigendecomposition(biasedResult.K, true);
	MatrixXd vecs(eig.vec.real().leftCols(config.nDim));
	VectorXd s0(eig.eig.real().segment(0, config.nDim));
	auto result = optimizeMatrix(s0,
		DebiasObjective(cons, vecs),
		DebiasGradient(cons, vecs),
		config.minDelta,
		config.maxIter,
		config.verbose);

	EmbedResult er;
	er.K = vecs * result.answer.asDiagonal() * vecs.transpose();
	er.loss = result.objective;
	return er;
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
