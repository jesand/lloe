/// @file  embed_cmp_with_gndms.cpp
/// @brief Implements Generalized Non-metric Multidimensional Scaling.

#include <ogt/embed/cmp.hpp>
#include <ogt/embed/dlib_opt.hpp>
#include <ogt/linalg/linalg.hpp>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using OGT_NAMESPACE::linalg::dist;
using OGT_NAMESPACE::linalg::projectOntoPSDCone;
using OGT_NAMESPACE::linalg::sqDistFromKernel;
using std::max;
using std::vector;

namespace OGT_NAMESPACE {
namespace embed {

/// The data container for the GNMDS objective information
struct GNMDSBase {
	GNMDSBase(const vector<CmpConstraint>& cons, VectorXd& slack, double lambda)
		: cons(cons), slack(slack), lambda(lambda) {}
	const vector<CmpConstraint>& cons;
	VectorXd& slack; // calculated in objective, shared with gradient
	const double lambda;
};

/// The GNMDS objective for kernel learning
struct GNMDSObjectiveK : GNMDSBase {

	/// Constructor.
	GNMDSObjectiveK(const vector<CmpConstraint>& cons, VectorXd& slack,
		double lambda)
		: GNMDSBase(cons, slack, lambda) {}

	/// Calculate loss
	double operator()(const MatrixXd& K) const {

		// Update the slack variables
		for (size_t ii = 0; ii < cons.size(); ii++) {
			const auto& con = cons[ii];
			double dab = sqDistFromKernel(K, con.a, con.b);
			double dac = sqDistFromKernel(K, con.a, con.c);
			slack[ii] = max(0.0, dab + 1 - dac);
		}

		// Compute the cost function
		return lambda * slack.sum() + (1 - lambda) * K.trace();
	}
};

/// The GNMDS gradient for kernel learning
struct GNMDSGradientK : GNMDSBase {

	/// Constructor.
	GNMDSGradientK(const vector<CmpConstraint>& cons, VectorXd& slack,
		double lambda)
		: GNMDSBase(cons, slack, lambda)
		, oldSlack(VectorXd::Zero(cons.size()))
	{}

	/// Calculate the gradient.
	void operator()(const MatrixXd& /*K*/, MatrixXd& grad) {

		// Update the gradient, relying on objective to have updated the slack
		// variables from K
		for (size_t ii = 0; ii < cons.size(); ii++) {
			const auto& con = cons[ii];
			if (oldSlack(ii) > 0 && slack(ii) == 0) {
				grad(con.a, con.b) += 2 * lambda;
				grad(con.b, con.a) += 2 * lambda;
				grad(con.a, con.c) -= 2 * lambda;
				grad(con.c, con.a) -= 2 * lambda;
				grad(con.b, con.b) -= lambda;
				grad(con.c, con.c) += lambda;
			} else if (oldSlack(ii) == 0 && slack(ii) > 0) {
				grad(con.a, con.b) -= 2 * lambda;
				grad(con.b, con.a) -= 2 * lambda;
				grad(con.a, con.c) += 2 * lambda;
				grad(con.c, con.a) += 2 * lambda;
				grad(con.b, con.b) += lambda;
				grad(con.c, con.c) -= lambda;
			}
		}
		oldSlack = slack;
	}

	/// The slack variables from the previous round
	VectorXd oldSlack;
};

/// The GNMDS objective for direct embedding learning
struct GNMDSObjectiveX : GNMDSBase {

	/// Constructor.
	GNMDSObjectiveX(const vector<CmpConstraint>& cons, VectorXd& slack,
		double lambda)
		: GNMDSBase(cons, slack, lambda) {}

	/// Calculate loss
	double operator()(const MatrixXd& X) const {

		// Calculate new slack variables
		for (size_t ii = 0; ii < cons.size(); ii++) {
			const auto& con = cons[ii];
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			slack[ii] = max(0.0, dab + 1 - dac);
		}

		// Compute the cost function
		return slack.sum() + lambda * X.cwiseProduct(X).sum();
	}
};

/// The GNMDS gradient for kernel learning
struct GNMDSGradientX : GNMDSBase {

	/// Constructor.
	GNMDSGradientX(const vector<CmpConstraint>& cons, VectorXd& slack,
		double lambda)
		: GNMDSBase(cons, slack, lambda)
	{}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) {

		// Update the gradient, relying on objective to have updated the slack
		// variables from X
		for (size_t ii = 0; ii < cons.size(); ii++) {
			const auto& con = cons[ii];
			if (slack(ii) > 0) {
				grad.row(con.a) += 2 * (X.row(con.a) - X.row(con.b));
				grad.row(con.a) -= 2 * (X.row(con.a) - X.row(con.c));
				grad.row(con.b) -= 2 * (X.row(con.a) - X.row(con.b));
				grad.row(con.c) += 2 * (X.row(con.a) - X.row(con.c));
			}
		}
		grad += 2 * lambda * X;
	}
};

/// Embeds a dataset by recovering a distance kernel using Generalized
/// non-metric multidimensional scaling.
EmbedResult embedCmpWithGNMDSForK(vector<CmpConstraint> cons,
	const MatrixXd& K0, double lambda, EmbedConfig config) {
	VectorXd slack = VectorXd::Zero(cons.size());
	auto result = optimizeMatrixWithAdjustment(K0,
		GNMDSObjectiveK(cons, slack, lambda),
		GNMDSGradientK(cons, slack, lambda),
		[](auto m) { return projectOntoPSDCone(m, true); },
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Generalized non-metric multidimensional scaling.
EmbedResult embedCmpWithGNMDSForX(vector<CmpConstraint> cons,
	const MatrixXd& X0, double lambda, EmbedConfig config) {
	VectorXd slack = VectorXd::Zero(cons.size());
	auto result = optimizeMatrix(X0,
		GNMDSObjectiveX(cons, slack, lambda),
		GNMDSGradientX(cons, slack, lambda),
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
