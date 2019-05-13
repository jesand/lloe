/// @file  cmp.hpp
/// @brief Declarations for embedding methods using classic ordinal constraints

#pragma once
#ifndef OGT_EMBED_CMP_HPP
#define OGT_EMBED_CMP_HPP

#include <ogt/config.hpp>
#include <ogt/core/oracle.hpp>
#include <ogt/embed/embed.hpp>
#include <Eigen/Dense>
#include <vector>

namespace OGT_NAMESPACE {
namespace embed {

/// An ordinal constraint specifying that dist(a,b) < dist(c,d).
struct CmpConstraint {

	/// Default constructor.
	CmpConstraint() : a(0), b(0), c(0), d(0) {}

	/// Constructor with four points
	CmpConstraint(size_t a, size_t b, size_t c, size_t d)
		: a(a), b(b), c(c), d(d) {}

	/// Constructor with three points: dist(a,b) < dist(a,c).
	CmpConstraint(size_t a, size_t b, size_t c)
		: CmpConstraint(a, b, c, a) {}

	/// Constructor from an oracle comparison
	CmpConstraint(const OGT_NAMESPACE::core::CmpOutcome& cmp)
		: CmpConstraint(
			cmp.a,
			cmp.cmp == OGT_NAMESPACE::core::AB_LT_AC ? cmp.b : cmp.c,
			cmp.cmp == OGT_NAMESPACE::core::AB_LT_AC ? cmp.c : cmp.b) {}

	/// The objects being constrained
	size_t a, b, c, d;
};

/// Embeds a dataset by recovering a dissimilarity kernel using the Crowd Kernel
/// embedding algorithm.
/// The parameter lambda effects the scaling of the embedding.
///
/// Citation: O. Tamuz, C. Liu, S. Belongie, O. Shamir, and A. T. Kalai,
/// "Adaptively Learning the Crowd Kernel," International Conference on Machine
/// Learning, 2011.
EmbedResult embedCmpWithCKForK(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, double lambda, EmbedConfig config);

/// Embeds a dataset using the Crowd Kernel embedding algorithm.
/// The parameter lambda effects the scaling of the embedding.
///
/// Citation: O. Tamuz, C. Liu, S. Belongie, O. Shamir, and A. T. Kalai,
/// "Adaptively Learning the Crowd Kernel," International Conference on Machine
/// Learning, 2011.
EmbedResult embedCmpWithCKForX(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, EmbedConfig config);

/// Embeds a dataset by recovering a dissimilarity kernel using Generalized
/// Non-metric Multidimensional Scaling.
/// The lambda parameter specifies the amount of regularization to apply to
/// keep the rank small.
/// An embedding can be recovered from the kernel using, for instance,
/// embeddingFromKernelSVD().
///
/// Citation: S. Agarwal, J. Wills, and L. Cayton, "Generalized non-metric
/// multidimensional scaling,"" International Conference on Machine Learning,
/// 2007.
EmbedResult embedCmpWithGNMDSForK(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, double lambda, EmbedConfig config);

/// Embeds a dataset using Generalized Non-metric Multidimensional Scaling.
/// The lambda parameter specifies the amount of regularization to apply to
/// keep the rank small.
///
/// Citation: S. Agarwal, J. Wills, and L. Cayton, "Generalized non-metric
/// multidimensional scaling,"" International Conference on Machine Learning,
/// 2007.
EmbedResult embedCmpWithGNMDSForX(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, EmbedConfig config);

/// Embeds a dataset using Soft Ordinal Embedding.
/// This method uses the margin parameter to set the scale. A default value of
/// 0.1 will be used if no margin is provided.
///
/// Citation: Y. Terada and U. von Luxburg, "Local ordinal embedding," presented
/// at the Proceedings of the 31st International Conference on Machine Learning,
/// 2014.
EmbedResult embedCmpWithSOE(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, EmbedConfig config);

/// Embeds using a weighted variant of Soft Ordinal Embedding. The loss function
/// is modified so that each constraint has a corresponding weight which is
/// multiplied to the loss incurred when that constraint is violated.
/// It is assumed, but not verified, that the weights are non-negative and sum
/// to one.
EmbedResult embedCmpWithSOEWeighted(std::vector<CmpConstraint> cons,
	const Eigen::VectorXd &weights, const Eigen::MatrixXd &X0,
	EmbedConfig config);

/// Embeds a point to satisfy a ranking of some set of fixed points, as much
/// as possible.
/// rank is an ordered list of rows from X. The resulting row in EmbedResult.X
/// gives a position which satisfies this ranking, if possible.
EmbedResult embedPtRankingWithSOE(const std::vector<size_t> &rank,
	const Eigen::MatrixXd& X, const Eigen::VectorXd& pos0, EmbedConfig config);

/// Embeds a dataset using Local Ordinal Embedding.
/// This method uses the margin parameter to set the scale. A default value of
/// 0.1 will be used if no margin is provided.
///
/// Citation: Y. Terada and U. von Luxburg, "Local ordinal embedding," presented
/// at the Proceedings of the 31st International Conference on Machine Learning,
/// 2014.
EmbedResult embedKnnWithLOE(const Eigen::MatrixXi& knn,
	const Eigen::MatrixXd& X0, EmbedConfig config);

/// Pick an optimal SOE scale parameter for the embedding.
double fitSOEScale(std::vector<CmpConstraint> cons, const Eigen::MatrixXd& X);

/// Embeds a dataset by recovering a dissimilarity kernel using Stochastic
/// Triplet Embedding.
/// The lambda parameter specifies the degree of regularization.
///
/// Citation: L. Van der Maaten and K. Weinberger, "Stochastic triplet
/// embedding," 2012 IEEE International Workshop on Machine Learning for Signal
/// Processing (MLSP), 2012.
EmbedResult embedCmpWithSTEForK(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& K0, double lambda, EmbedConfig config);

/// Embeds a dataset using Stochastic Triplet Embedding.
/// The lambda parameter specifies the degree of regularization.
///
/// Citation: L. Van der Maaten and K. Weinberger, "Stochastic triplet
/// embedding," 2012 IEEE International Workshop on Machine Learning for Signal
/// Processing (MLSP), 2012.
EmbedResult embedCmpWithSTEForX(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, EmbedConfig config);

/// Embeds a dataset using t-Distributed Stochastic Triplet Embedding.
/// The lambda parameter specifies the degree of regularization.
/// The alpha parameter indicates the degrees of freedom for the Student's-t
/// distribution.
///
/// Citation: L. Van der Maaten and K. Weinberger, "Stochastic triplet
/// embedding," 2012 IEEE International Workshop on Machine Learning for Signal
/// Processing (MLSP), 2012.
EmbedResult embedCmpWithTSTE(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, double alpha, EmbedConfig config);

/// Embeds a dataset using Rank-d Projected Gradient Descent (PGD), adapted
/// to work directly on a n x d matrix X rather than to project after each step.
///
/// Citation: L. Jain, K. Jamieson, and R. Nowak, "Finite Sample Prediction and
/// Recovery Bounds for Ordinal Embedding," NIPS, 2016.
EmbedResult embedCmpWithPGDForX(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, EmbedConfig config);

/// Embeds a dataset using Rank-d Projected Gradient Descent (PGD).
/// This is the original version of the algorithm.
/// After each learning step, we project to the nearest rank d Gram matrix.
///
/// Citation: L. Jain, K. Jamieson, and R. Nowak, "Finite Sample Prediction and
/// Recovery Bounds for Ordinal Embedding," NIPS, 2016.
EmbedResult embedCmpWithPGDForK(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, EmbedConfig config);

/// Embeds a dataset using Nuclear Norm Projected Gradient Descent.
/// After each learning step, we project onto the nuclear norm ball, which has
/// the effect of minimizing the rank.
///
/// Citation: L. Jain, K. Jamieson, and R. Nowak, "Finite Sample Prediction and
/// Recovery Bounds for Ordinal Embedding," NIPS, 2016.
EmbedResult embedCmpWithNNPGD(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, EmbedConfig config);

/// Embeds a dataset using Nuclear Norm Projected Gradient Descent Debiased.
/// This is similar to embedCmpWithNNPGD() but with an extra debiasing step.
/// This debiasing improves accuracy by rescaling the non-zero eigenvalues
/// to prevent them from shrinking toward zero. It is the default behavior.
///
/// Citation: L. Jain, K. Jamieson, and R. Nowak, "Finite Sample Prediction and
/// Recovery Bounds for Ordinal Embedding," NIPS, 2016.
EmbedResult embedCmpWithNNPGDDebiased(std::vector<CmpConstraint> cons,
	const Eigen::MatrixXd& X0, double lambda, EmbedConfig config);

/// Compute a kernel matrix where K_ij is derived by how similarly points i and
/// j are ranked by the provided comparisons.
/// The parameters k1 and k2 indicate how much of the two possible kernels to
/// mix into the final output; k1 corresponds to an anchor-based kernel while k2
/// corresponds to a tail-based kernel.
///
/// Citation: M. Kleindessner and U. von Luxburg, “Kernel functions based on
/// triplet similarity comparisons,” arXiv.org, vol. stat.ML. 28-Jul-2016.
EmbedResult embedCmpWithTauForK(std::vector<CmpConstraint> cons, double k1,
	double k2);

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_CMP_HPP */
