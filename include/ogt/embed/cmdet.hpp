/// @file  cmdet.hpp
/// @brief Declarations for methods based on Cayley-Menger determinants.

#pragma once
#ifndef OGT_EMBED_CMDET_HPP
#define OGT_EMBED_CMDET_HPP

#include <ogt/config.hpp>
#include <Eigen/Dense>
#include <memory>

namespace OGT_NAMESPACE {
namespace embed {

/// Captures a set of Cayley-Menger reference vertices which can be used to find
/// pairwise distances between points, or to embed the set.
///
/// This method is based on:
/// [1]	M. J. Sippl and H. A. Scheraga, "Cayley-Menger coordinates.,"
///     Proceedings of the National Academy of Sciences, Apr. 1986.
struct CMReference {

	/// Construct a concrete instance, given all pairwise squared Euclidean
	/// distances between the reference vertices.
	static std::shared_ptr<CMReference> Create(Eigen::MatrixXd refSqDists);

	/// Virtual destructor.
	virtual ~CMReference() = default;

	/// Find the squared Euclidean distance between two points with the given
	/// squared Euclidean distances to the reference vertices.
	virtual double sqDist(Eigen::VectorXd v1, Eigen::VectorXd v2) = 0;
};

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_CMDET_HPP */
