/// @file  sphere.hpp
/// @brief Declarations for methods to find sphere intersections.
///
/// The intersection of d spheres in R^d is either zero, one, or two points.
/// More than d spheres can only intersect in zero or one points.
/// The methods here provide alternative ways to compute the intersection.
///
/// sphere_intersection_gauss() is the fastest but least reliable. It requires
/// exactly d spheres, expressed as centerpoints in A and radii in r, and will
/// find an exact solution only when A is invertible. In particular, none of the
/// centerpoints can be at the origin, as this would cause a zero column in A.
///
/// sphere_intersection_gauss() and sphere_intersection_orth() are both O(d^3),
/// but sphere_intersection_orth() takes roughly twice as long. Its advantage is
/// that sphere_intersection_orth() can handle non-invertible matrices A.
///
/// The final method, sphere_intersection_opt(), can also find a point close to
/// all the spheres in the case that they do not intersect.
/// It first runs sphere_intersection_orth(), and if the spheres intersect it
/// will return that solution. If they do not intersect, it will run a
/// Newton-Rhapson method to optimize a least-squares objective to choose the
/// unique point "closest" to all the spheres, in the sense that it minimizes
/// the sum of the squared Euclidean distances to the sphere boundaries.
/// The optimization method generally converges in just a few iterations, so
/// in practice sphere_intersection_opt() is generally not much slower than
/// sphere_intersection_orth().
///
/// Ref: I. D. Coope, “Reliable computation of the points of intersection of $n$
///   spheres in $R^n$,” ANZIAM Journal, vol. 42, no. 0, pp. 461–477, Dec. 2000.

#pragma once
#ifndef OGT_EMBED_SPHERE_HPP
#define OGT_EMBED_SPHERE_HPP

#include <ogt/config.hpp>
#include <vector>
#include <Eigen/Dense>

namespace OGT_NAMESPACE {
namespace embed {

/// Find the two points where n spheres intersect in R^n by Gaussian
/// elimination.
/// center is nxn; its columns are the locations of the sphere centers.
/// radius is nx1; d_i is the radius of the i'th sphere in center.
/// 2*eps is the smallest permissible distance between intersection points to
/// consider them distinct.
/// If only one intersection point exists, the second vector will have size 0.
/// If neither intersection point exists, both vectors will have size 0.
///
/// Throws std::invalid_argument if the matrix dimensions are incorrect.
///
/// @see sphere_intersection_orth(), sphere_intersection_opt()
std::vector<Eigen::VectorXd> sphere_intersection_gauss(
	const Eigen::MatrixXd &center, const Eigen::VectorXd& radius, double eps);

/// Find the two points where n spheres intersect in R^n by an orthogonal
/// transformation. This method is more robust than sphere_intersection_gauss,
/// but takes up to twice as long.
///
/// Throws std::invalid_argument if the matrix dimensions are incorrect.
///
/// @see sphere_intersection_gauss(), sphere_intersection_opt()
std::vector<Eigen::VectorXd> sphere_intersection_orth(
	const Eigen::MatrixXd &center, const Eigen::VectorXd& radius, double eps);

/// Find points close to n or more spheres in n dimensions. If the spheres
/// intersect, the interesection points are returned. Otherwise, an optimization
/// based approach is taken to find a point minimizing the distance to each
/// sphere.
///
/// Throws std::invalid_argument if the matrix dimensions are incorrect.
/// Throws ogt::EmbedErr if the embedding fails.
///
/// @see sphere_intersection_gauss(), sphere_intersection_orth()
std::vector<Eigen::VectorXd> sphere_intersection_opt(
	const Eigen::MatrixXd &center, const Eigen::VectorXd& radius, double eps,
	double min_delta, size_t max_iter, bool verbose);

/// Find a point within a certain margin of n or more spheres in n dimensions.
/// If the spheres intersect, an interesection point is returned.
/// Otherwise, a point is found to minimize the distance to each spherical
/// shell, defined as points with distances radius +/- margin of the center.
///
/// Throws std::invalid_argument if the matrix dimensions are incorrect.
/// Throws ogt::EmbedErr if the embedding fails.
///
/// @see sphere_intersection_gauss(), sphere_intersection_orth()
std::vector<Eigen::VectorXd> sphere_intersection_margin(
	const Eigen::MatrixXd &center, const Eigen::VectorXd& radius,
	const Eigen::VectorXd& margin, double eps,
	double min_delta, size_t max_iter, bool verbose);

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_SPHERE_HPP */
