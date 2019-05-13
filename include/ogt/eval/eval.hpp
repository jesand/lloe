/// @file  eval.hpp
/// @brief Declarations for evaluation methods

#pragma once
#ifndef OGT_EVAL_EVAL_HPP
#define OGT_EVAL_EVAL_HPP

#include <ogt/config.hpp>
#include <ogt/util/chain_merge.hpp>
#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace OGT_NAMESPACE {
namespace eval {

/// Calculate raw stress. This is the sum of the squared error of the distances.
/// It takes its minimum at zero, which indicates a perfect embedding.
/// If requested, the estimated distances can first be scaled by the optimal
/// constant to minimize stress.
/// The scaled versions of these values can be used to compare embeddings of the
/// same dataset, but as they are not normalized they cannot be compared across
/// datasets.
///
/// The unscaled version is referred to as \sigma_r in:
/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
///     and applications. Springer.
double mdsStress(const Eigen::MatrixXd& dhat, const Eigen::MatrixXd& dtrue,
	bool scaled = false);

/// Calculate normed stress. This is a normalized version of mdsStress(), which
/// can thus be compared between different datasets.
/// It takes its minimum at zero, which indicates a perfect embedding.
///
/// The unscaled version is referred to as Stress-1 or \sigma_1 in:
/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
///     and applications. Springer.
double mdsNormedStress(const Eigen::MatrixXd& dhat,
	const Eigen::MatrixXd& dtrue, bool scaled = false);

/// Calculate normed rank stress. That is, the normalized sum-squared amount by
/// which each similarity constraint is violated.
/// This essentially treats dist(i,j) as the target distance for dist(k,l)
/// if dist(i,j) immediately precedes dist(k,l) in the partial order of
/// distances.
double mdsNormedRankStress(const Eigen::MatrixXd& dhat,
	const OGT_NAMESPACE::util::ChainMerge& order);

/// Calculate the root mean squared error of the distances, after finding the
/// scaling which minimizes this error.
/// In other words, we report RMSE for the best linear fit between the
/// distance matrices.
double distRmse(const Eigen::MatrixXd& dhat, const Eigen::MatrixXd& dtrue);

/// Calculate the Kendall's tau-b of two distance vectors to other points.
/// Does so in O(n log n) time, using
/// Knight's Algorithm](http://adereth.github.io/blog/2013/10/30/efficiently-computing-kendalls-tau/).
double kendallTau(const Eigen::VectorXd& d1, const Eigen::VectorXd& d2);

/// Calculate the mean Kendall's tau-b of rankings by each row.
double meanKendallTau(const Eigen::MatrixXd& dhat,
	const Eigen::MatrixXd& dtrue);

/// Calculate the weighted Kendall's tau of two vectors of numbers.
/// The tau value is given for vectors r and s. The function w provides
/// a weight for each rank in the list.
///
/// [1] S. Vigna, A Weighted Correlation Index for Rankings with Ties. WWW, 2015
double weightedTau(const Eigen::VectorXd& r, const Eigen::VectorXd& s,
	std::function<double(size_t /* rank */)> w);

/// Calculate the weighted Kendall's tau of two vectors of numbers.
/// The tau value is given for vectors r and s. The functions w1 and w2 provide
/// weights for each rank in the list.
///
/// [1] S. Vigna, A Weighted Correlation Index for Rankings with Ties. WWW, 2015
double weightedTau(const Eigen::VectorXd& r, const Eigen::VectorXd& s,
	std::function<double(size_t /* rank */)> w1,
	std::function<double(size_t /* rank */)> w2);

/// The hyperbolic tau: weightedTau with rank weight 1/(1 + r).
double hyperbolicTau(const Eigen::VectorXd& r, const Eigen::VectorXd& s);

/// Calculate the mean hyperbolic tau of rankings by each row.
double meanHyperbolicTau(const Eigen::MatrixXd& dhat,
	const Eigen::MatrixXd& dtrue);

} // end namespace eval
} // end namespace OGT_NAMESPACE
#endif /* OGT_EVAL_EVAL_HPP */
