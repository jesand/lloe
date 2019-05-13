/// @file  embed.hpp
/// @brief Declarations for embedding methods

#pragma once
#ifndef OGT_EMBED_EMBED_HPP
#define OGT_EMBED_EMBED_HPP

#include <ogt/config.hpp>
#include <ogt/core/oracle.hpp>
#include <Eigen/Dense>
#include <map>
#include <memory>

namespace OGT_NAMESPACE {
namespace embed {


/// Configuration for an embedding method. Not all settings are used for all
/// methods.
struct EmbedConfig {

	/// Default constructor
	EmbedConfig() : nDim(0), minDelta(0), maxIter(0), verbose(false),
		useOracle(false), minSphereDiff(0), skipBad(false), margin(0) {}

	/// The number of dimensions into which to embed.
	size_t nDim;

	/// Convergence threshold for improvements to the optimization objective.
	double minDelta;

	/// Maximum number of optimization iterations to run.
	size_t maxIter;

	/// Whether to print extra status messages to STDOUT.
	bool verbose;

	/// A comparison oracle; ignored unless useOracle is true.
	std::shared_ptr<OGT_NAMESPACE::core::Oracle> oracle;

	/// Whether to use the oracle to tighten constraints, when possible.
	bool useOracle;

	/// Fixed point positions; treat these as constants.
	std::map<size_t, Eigen::VectorXd> fixed;

	/// embedTrianglesWithSpheres(): When two solutions are found which differ
	///   by this amount, treat them as one solution.
	double minSphereDiff;

	/// embedTrianglesWithSpheres(): If embedding any particular point fails,
	/// place it at (NaN, NaN, ...).
	bool skipBad;

	/// embedTrianglesWithOpt(): The extra margin by which constraints should
	/// be satisfied. Sets the embedding scale.
	double margin;
};

/// The result of an embedding method
struct EmbedResult {

	/// Constructor
	EmbedResult() : loss(0), iter(0), numSkipped(0) {}

	/// The embedding
	Eigen::MatrixXd X;

	/// The kernel (the distance matrix)
	Eigen::MatrixXd K;

	/// The loss achieved, if any
	double loss;

	/// The number of iterations used, if any
	size_t iter;

	/// The number of points which were skipped, if any
	size_t numSkipped;
};

/// An exception thrown when embedding fails
struct EmbedErr : public std::runtime_error {

	/// Raise an error, optionally also printing it if running in verbose mode
	static void report(std::string message, bool verbose);

	/// Build an EmbedErr with the specified message.
	EmbedErr(std::string message)
		: std::runtime_error("ogt::EmbedErr: " + message) {
	}
};

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_EMBED_HPP */
