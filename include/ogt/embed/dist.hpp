/// @file  dist.hpp
/// @brief Declarations for embedding methods using pairwise distances

#pragma once
#ifndef OGT_EMBED_DIST_HPP
#define OGT_EMBED_DIST_HPP

#include <ogt/config.hpp>
#include <Eigen/Dense>

namespace OGT_NAMESPACE {
namespace embed {

/// Produce an embedding of the specified dimensionality from the given
/// Euclidean distance matrix.
///
/// Uses Classical MDS, as described in:
/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
///     and applications. Springer.
Eigen::MatrixXd embedDistWithDims(const Eigen::MatrixXd& dist, size_t nDim);

/// Produce an embedding from the given Euclidean distance matrix.
/// The natural dimensionality is estimated, using all eigenvalues greater than
/// or equal to the (non-negative) threshold lambda.
///
/// Uses Classical MDS, as described in:
/// [1] I. Borg & P. Groenen (1997): Modern multidimensional scaling: theory
///     and applications. Springer.
Eigen::MatrixXd embedDist(const Eigen::MatrixXd& dist,
	double lambda = 1e-12);

/// Produce an embedding using Euclidean distances to a subset of the set.
/// The matrix dist must be a n x (d+1) Euclidean distance matrix, where each
/// column represents the distance to some reference vertex.
/// The reference vertices must be the first (d+1) rows.
/// The produced embedding will be in R^maxDim, or in fewer dimensions if the
/// distances between reference vertices are not large enough to justify the
/// full dimensionality.
///
/// Implements:
/// [1]	M. J. Sippl and H. A. Scheraga, "Solution of the embedding problem and
///     decomposition of symmetric matrices.," Proceedings of the National
///     Academy of Sciences, Apr. 1985.
Eigen::MatrixXd embedDistWithCMReference(const Eigen::MatrixXd& dist,
	size_t maxDim);

/// Produce an embedding into the target dimensionality, attempting to preserve
/// the total ordering of pairwise distances.
/// An ordinal matrix is produced from the given distance matrix, and an
/// eigendecomposition is used to embed the points.
///
/// Implements method from:
/// [1] Dattorro, Jon, "Convex Optimization and Euclidean Distance Geometry,"
///     Meboo Publishing, 2016.
Eigen::MatrixXd embedDistWithOrdEig(const Eigen::MatrixXd& dist, size_t nDim);


} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_DIST_HPP */
