/// @file  oracle_inference.hpp
/// @brief Declarations related to inferring triples from known triples.

#pragma once
#ifndef OGT_CORE_ORACLE_INFERENCE_HPP
#define OGT_CORE_ORACLE_INFERENCE_HPP

#include <ogt/config.hpp>
#include <ogt/core/oracle.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace OGT_NAMESPACE {
namespace core {

// Forward declarations
struct Collection;
class Traversal;

/// An object to infer triples based on known triples.
struct InferenceOracle : public Oracle {

	/// Create an inference oracle based on rank order agreement.
	static std::shared_ptr<InferenceOracle> CreateRankOracle(
		std::shared_ptr<Collection> coll);

	/// Create an inference oracle based on ordinal vector projection.
	static std::shared_ptr<InferenceOracle> CreateOrdinalOracle(
		std::shared_ptr<Collection> coll);

	/// Create an inference oracle based on exact vector projection.
	static std::shared_ptr<InferenceOracle> CreateDistanceOracle(
		std::shared_ptr<Collection> coll, const Eigen::MatrixXd& points);

	/// Ctor
	InferenceOracle(std::shared_ptr<Collection> coll) : coll(coll) {}

	/// Dtor
	virtual ~InferenceOracle() = default;

	/// Visit an anchor, permitting its ranking to be used for inference.
	virtual void visitAnchor(size_t anchor) = 0;

	/// Decide whether a triple is correct. A positive value is a "yes,"
	/// a negative value is a "no," and a zero represents abstention.
	/// Values with larger absolute values indicate greater confidence.
	virtual double isTriple(size_t a, size_t b, size_t c) = 0;

	/// Attempt to sort a list of objects using inferred triples.
	virtual void sort(size_t head, std::vector<size_t>& objects) = 0;

	/// Train the oracle by adding anchors in the specified traversal order.
	virtual void train(Traversal order, bool evalEach);

protected:

	/// Ask whether x sorts before y under projection pq.
	/// The return values are the same as for isTriple: positive is "yes,"
	/// negative is "no," zero is abstention, and magnitude indicates
	/// confidence.
	virtual double isXBeforeY(size_t p, size_t q, size_t x, size_t y) = 0;

	std::shared_ptr<Collection> coll;
};

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_ORACLE_INFERENCE_HPP */
