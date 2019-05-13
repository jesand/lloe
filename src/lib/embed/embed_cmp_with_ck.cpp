/// @file  embed_cmp_with_soe.cpp
/// @brief Implements Soft Ordinal Embedding.

#include <ogt/embed/cmp.hpp>
#include <ogt/embed/dlib_opt.hpp>
#include <ogt/linalg/linalg.hpp>
#include <limits>

using Eigen::MatrixXd;
using OGT_NAMESPACE::linalg::projectOntoPSDCone;
using OGT_NAMESPACE::linalg::sqDistFromKernel;
using std::vector;
using std::max;

namespace OGT_NAMESPACE {
namespace embed {

/// The data container for the CK objective information
struct CKBase {

	/// Constructor
	CKBase(const vector<CmpConstraint>& cons, double lambda)
		: cons(cons), lambda(lambda) {}

	const vector<CmpConstraint>& cons;
	const double lambda;
};

/// The CK objective for X
struct CKObjectiveX : CKBase {

	/// Constructor
	CKObjectiveX(const vector<CmpConstraint>& cons, double lambda)
		: CKBase(cons, lambda) {}

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
			double num = max(minval, lambda + dac);
			double denom = max(minval, 2 * lambda + dab + dac);
			loss -= log(num) - log(denom);
		}
		return loss;
	}
};

/// The gradient of the CK optimization objective for X
struct CKGradientX : CKBase {

	/// Constructor.
	CKGradientX(const vector<CmpConstraint>& cons, double lambda)
		: CKBase(cons, lambda) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) const {
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			const auto xa = X.row(con.a);
			const auto xb = X.row(con.b);
			const auto xc = X.row(con.c);
			double dab = (xa - xb).squaredNorm();
			double dac = (xa - xc).squaredNorm();
			double num = max(minval, lambda + dac);
			double denom = max(minval, 2 * lambda + dab + dac);

			// grad.row(con.a) -= (2 * num * (xa - xb)) / (denom * denom);
			// grad.row(con.b) += (2 * num * (xa - xb)) / (denom * denom);
			// grad.row(con.a) -= (2 * num * (xa - xc)) / (denom * denom);
			// grad.row(con.c) += (2 * num * (xa - xc)) / (denom * denom);
			// grad.row(con.a) += (2 * (xa - xc)) / denom;
			// grad.row(con.c) -= (2 * (xa - xc)) / denom;

			// grad.row(con.a) -= (2 / num) * (xa - xc);
			// grad.row(con.a) += (2 / denom) * ((xa - xb) + (xa - xc));
			// grad.row(con.b) -= (2 / denom) * (xa - xb);
			// grad.row(con.c) += (2 / num) * (xa - xc);
			// grad.row(con.c) += (2 / denom) * (xa - xc);

			// dC(:,d) = dC(:,d) + accumarray(triplets(:,1),  
			// 	2 ./ nom .*  (X(triplets(:,1), d) - X(triplets(:,3), d)) - ...
			// 	2 ./ den .* ((X(triplets(:,1), d) - X(triplets(:,2), d)) + ...
			// 	(X(triplets(:,1), d) - X(triplets(:,3), d))), [N 1]);
			grad.row(con.a) -= (2 / num) * (xa - xc);
			grad.row(con.a) += (2 / denom) * ((xa - xb) + (xa - xc));

			// dC(:,d) = dC(:,d) + accumarray(triplets(:,2),
			// 	2 ./ den .*  (X(triplets(:,1), d) - X(triplets(:,2), d)),  [N 1]);
			grad.row(con.b) -= (2 / denom) * (xa - xb);
			
			// dC(:,d) = dC(:,d) + accumarray(triplets(:,3),
			// 	-2 ./ nom .* (X(triplets(:,1), d) - X(triplets(:,3), d)) + ...
			// 	2 ./ den .* (X(triplets(:,1), d) - X(triplets(:,3), d)), [N 1]);
			grad.row(con.c) += (2 / num) * (xa - xc);
			grad.row(con.c) -= (2 / denom) * (xa - xc);

			// (then flip sign of dC)
		}
	}
};

/// The CK objective for K
struct CKObjectiveK : CKBase {

	/// Constructor
	CKObjectiveK(const vector<CmpConstraint>& cons, double lambda)
		: CKBase(cons, lambda) {}

	/// Calculate loss
	double operator()(const MatrixXd& K) const {
		double loss = 0;
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			double dab = sqDistFromKernel(K, con.a, con.b);
			double dac = sqDistFromKernel(K, con.a, con.c);
			double num = max(minval, lambda + dac);
			double denom = max(minval, 2 * lambda + dab + dac);
			loss -= log(num) - log(denom);
		}
		return loss;
	}
};

/// The gradient of the CK optimization objective for K
struct CKGradientK : CKBase {

	/// Constructor.
	CKGradientK(const vector<CmpConstraint>& cons, double lambda)
		: CKBase(cons, lambda) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& K, MatrixXd& grad) const {
		const double minval = std::numeric_limits<double>::min();
		for (const auto& con : cons) {
			double dab = sqDistFromKernel(K, con.a, con.b);
			double dac = sqDistFromKernel(K, con.a, con.c);
			double num = max(minval, lambda + dac);
			double denom = max(minval, 2 * lambda + dab + dac);

			grad(con.a, con.a) -= (1 / num) - (2 / denom);
			grad(con.b, con.b) += 1 / denom;
			grad(con.c, con.c) -= (1 / num) - (1 / denom);
			grad(con.a, con.b) -= 2 / denom;
			grad(con.b, con.a) -= 2 / denom;
			grad(con.a, con.c) += (2 / num) - (2 / denom);
			grad(con.c, con.a) += (2 / num) - (2 / denom);
		}
	}
};

/// Embeds a dataset using the Crowd Kernel embedding algorithm.
EmbedResult embedCmpWithCKForX(vector<CmpConstraint> cons,
	const MatrixXd& X0, double lambda, EmbedConfig config) {
	auto result = optimizeMatrix(X0,
		CKObjectiveX(cons, lambda),
		CKGradientX(cons, lambda),
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset by recovering a dissimilarity kernel using the Crowd Kernel
/// embedding algorithm.
EmbedResult embedCmpWithCKForK(vector<CmpConstraint> cons,
	const MatrixXd& K0, double lambda, EmbedConfig config) {
	auto result = optimizeMatrixWithAdjustment(K0,
		CKObjectiveK(cons, lambda),
		CKGradientK(cons, lambda),
		[](auto m) { return projectOntoPSDCone(m, false); },
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
