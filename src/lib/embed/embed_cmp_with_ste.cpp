/// @file  embed_cmp_with_ste.cpp
/// @brief Implements Stochastic Triplet Embedding.

#include <ogt/embed/cmp.hpp>
#include <ogt/embed/dlib_opt.hpp>
#include <ogt/linalg/linalg.hpp>
#include <limits>
#include <cmath>

using Eigen::MatrixXd;
using OGT_NAMESPACE::linalg::projectOntoPSDCone;
using OGT_NAMESPACE::linalg::sqDistFromKernel;
using std::log;
using std::pow;
using std::vector;
using std::max;

namespace OGT_NAMESPACE {
namespace embed {

/// The data container for the STE objective information
struct STEBase {

	/// Constructor
	STEBase(const vector<CmpConstraint>& cons, double lambda)
		: cons(cons), lambda(lambda) {}

	const vector<CmpConstraint>& cons;
	const double lambda;
};

/// The STE objective for K
struct STEObjectiveK : STEBase {

	/// Constructor
	STEObjectiveK(const vector<CmpConstraint>& cons, double lambda)
		: STEBase(cons, lambda) {}

	/// Calculate loss
	double operator()(const MatrixXd& K) const {
		double loss = 0;
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			double dab = sqDistFromKernel(K, con.a, con.b);
			double dac = sqDistFromKernel(K, con.a, con.c);
			double lognum = -dab;
			double denom = max(minval, exp(-dab) + exp(-dac));
			loss -= lognum - log(denom);
		}
		return loss + lambda * K.trace();
	}
};

/// The gradient of the STE optimization objective for K
struct STEGradientK : STEBase {

	/// Constructor.
	STEGradientK(const vector<CmpConstraint>& cons, double lambda)
		: STEBase(cons, lambda) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& K, MatrixXd& grad) const {
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			double dab = sqDistFromKernel(K, con.a, con.b);
			double dac = sqDistFromKernel(K, con.a, con.c);
			double num = exp(-dab);
			double denom = exp(-dab) + exp(-dac);
			double prob = max(minval, exp(log(num) - log(denom)));

			grad(con.a, con.b) -= 2 * (1 - prob);
			grad(con.b, con.a) -= 2 * (1 - prob);
			grad(con.a, con.c) += 2 * (1 - prob);
			grad(con.c, con.a) += 2 * (1 - prob);
			grad(con.b, con.b) += 1 - prob;
			grad(con.c, con.c) -= 1 - prob;
		}
		grad *= (1 - lambda);
		grad += lambda * MatrixXd::Identity(grad.rows(), grad.cols());
	}
};

/// The STE objective for X
struct STEObjectiveX : STEBase {

	/// Constructor
	STEObjectiveX(const vector<CmpConstraint>& cons, double lambda)
		: STEBase(cons, lambda) {}

	/// Calculate loss
	double operator()(const MatrixXd& X) const {
		double loss = 0;
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			double lognum = -dab;
			double denom = max(minval, exp(-dab) + exp(-dac));
			loss -= lognum - log(denom);
		}
		return loss + lambda * X.squaredNorm();
	}
};

/// The gradient of the STE optimization objective for X
struct STEGradientX : STEBase {

	/// Constructor.
	STEGradientX(const vector<CmpConstraint>& cons, double lambda)
		: STEBase(cons, lambda) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) const {
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			double num = max(minval, exp(-dab));
			double denom = max(minval, exp(-dab) + exp(-dac));
			double prob = num / denom;

			grad.row(con.a) += 2 * (1 - prob) * ((xa - xb) - (xa - xc));
			grad.row(con.b) -= 2 * (1 - prob) * (xa - xb);
			grad.row(con.c) += 2 * (1 - prob) * (xa - xc);
		}
		grad += (2 * lambda) * X;
	}
};

/// The t-STE objective
struct TSTEObjective : STEBase {

	/// Constructor
	TSTEObjective(const vector<CmpConstraint>& cons, double lambda,
		double alpha)
		: STEBase(cons, lambda), alpha(alpha) {}

	/// Calculate loss
	double operator()(const MatrixXd& X) const {
		double loss = 0;
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			double tab = pow(1 + (dab / alpha), -(1 + alpha) / 2);
			double tac = pow(1 + (dac / alpha), -(1 + alpha) / 2);
			double num = max(minval, tab);
			double denom = num + max(minval, tac);
			loss -= log(num) - log(denom);
		}
		return loss + lambda * X.squaredNorm();
	}

	/// The degrees of freedom.
	const double alpha;
};

/// The gradient of the t-STE optimization objective
struct TSTEGradient : STEBase {

	/// Constructor.
	TSTEGradient(const vector<CmpConstraint>& cons, double lambda, double alpha)
		: STEBase(cons, lambda), alpha(alpha) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) const {
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			double tab = pow(1 + (dab / alpha), -(1 + alpha) / 2);
			double tac = pow(1 + (dac / alpha), -(1 + alpha) / 2);
			double num = max(minval, tab);
			double denom = num + max(minval, tac);
			double prob = exp(log(num) - log(denom));

			double scale = (1 - prob) * (1 + alpha);
			grad.row(con.a) += scale * (xa - xb) / (alpha + dab);
			grad.row(con.a) -= scale * (xa - xc) / (alpha + dac);
			grad.row(con.b) -= scale * (xa - xb) / (alpha + dab);
			grad.row(con.c) += scale * (xa - xc) / (alpha + dac);
		}
		grad += (2 * lambda) * X;
	}

	/// The degrees of freedom.
	const double alpha;
};

/// Embeds a dataset by recovering a dissimilarity kernel using Stochastic
/// Triplet Embedding.
EmbedResult embedCmpWithSTEForK(vector<CmpConstraint> cons,
	const MatrixXd& K0, double lambda, EmbedConfig config) {
	auto result = optimizeMatrixWithAdjustment(K0,
		STEObjectiveK(cons, lambda),
		STEGradientK(cons, lambda),
		[](auto m) { return projectOntoPSDCone(m, true); },
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Stochastic Triplet Embedding.
EmbedResult embedCmpWithSTEForX(vector<CmpConstraint> cons,
	const MatrixXd& X0, double lambda, EmbedConfig config) {
	auto result = optimizeMatrix(X0,
		STEObjectiveX(cons, lambda),
		STEGradientX(cons, lambda),
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using t-Distributed Stochastic Triplet Embedding.
EmbedResult embedCmpWithTSTE(vector<CmpConstraint> cons,
	const MatrixXd& X0, double lambda, double alpha, EmbedConfig config) {
	auto result = optimizeMatrixKeepMin(X0,
		TSTEObjective(cons, lambda, alpha),
		TSTEGradient(cons, lambda, alpha),
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
