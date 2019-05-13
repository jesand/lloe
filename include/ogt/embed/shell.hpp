/// @file  shell.hpp
/// @brief Declarations for methods to find spherical shell intersections.

#pragma once
#ifndef OGT_EMBED_SHELL_HPP
#define OGT_EMBED_SHELL_HPP

#include <ogt/config.hpp>
#include <vector>
#include <Eigen/Dense>

namespace OGT_NAMESPACE {
namespace embed {

/// Find a point within the intersection of n or more spherical shells in n
/// dimensions. If the shells do not intersect, a point is found which is as
/// close as possible to the shells using a least-squares objective.
///
/// Throws std::invalid_argument if the matrix dimensions are incorrect.
/// Throws ogt::EmbedErr if the embedding fails.
///
/// @see sphere_intersection_opt()
Eigen::VectorXd shell_intersection(const Eigen::MatrixXd &A,
	const Eigen::VectorXd& d, const Eigen::VectorXd& t, double eps,
	double min_delta, size_t max_iter, bool verbose);

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_SHELL_HPP */
