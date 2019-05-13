/// @file  triangle.hpp
/// @brief Declarations for methods based on triangle constraints

#pragma once
#ifndef OGT_EMBED_TRIANGLE_HPP
#define OGT_EMBED_TRIANGLE_HPP

#include <ogt/config.hpp>
#include <ogt/core/collection.hpp>
#include <ogt/embed/embed.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <vector>

using OGT_NAMESPACE::core::Collection;
using OGT_NAMESPACE::core::Oracle;
using OGT_NAMESPACE::core::Traversal;

namespace OGT_NAMESPACE {
namespace embed {

struct CmpConstraint; // Forward declaration

/// A triangle with two edges' lengths constrained to be scaled to within some
/// interval of the length of the third edge.
struct TriangleConstraint {

	/// Initializing constructor
	TriangleConstraint(size_t p, size_t x, size_t y, double xmin,
		double xmax, double ymin, double ymax)
		: p(p), x(x), y(y), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax)
	{
	}

	/// Initialize from a vector, e.g. loaded from .csv
	TriangleConstraint(const Eigen::VectorXd& vec)
		: p(vec(0)), x(vec(1)), y(vec(2)), xmin(vec(3)), xmax(vec(4))
		, ymin(vec(5)), ymax(vec(6))
	{
	}

	/// Elementwise equality test
	bool operator==(const TriangleConstraint& other) const {
		return p == other.p && x == other.x && y == other.y
			&& xmin == other.xmin && xmax == other.xmax && ymin == other.ymin
			&& ymax == other.ymax;
	}

	/// Tighten the constraint using the triangle inequality.
	///
	/// We can update xmin and ymin as follows.
	/// ~~~~
	/// xy <= px + py <= px + xy * ymax
	///    --> px >= xy(1 - ymax)
	///    --> px >= xy * max(xmin, 1 - ymax)
	///    --> py >= xy * max(ymin, 1 - xmax) (by symmetry)
	/// ~~~~
	void tighten();

	/// Rotate the constraint to use a different edge as the reference distance.
	/// If toX is true, the output's p will be the input's x point. Otherwise,
	/// the output's p will be the input's y point.
	///
	/// The update rule is derived as follows.
	/// ~~~~
	/// xmin <= px/xy <= xmax
	///     --> 1/xmax <= xy/px <= 1/xmin
	/// py <= ymax * xy
	///    <= ymax * px / xmin
	///    --> ymin / xmax <= py/px <= ymax / xmin
	/// ~~~~
	TriangleConstraint rotate(bool toX, double maxRatio = INFINITY) const;

	/// The point incident to the two constrained edges.
	size_t p;

	/// One of the points incident to the reference edge.
	size_t x;

	/// One of the points incident to the reference edge.
	size_t y;

	/// A lower bound on the ratio dist(p,x) / dist(x,y)
	double xmin;

	/// An upper bound on the ratio dist(p,x) / dist(x,y)
	double xmax;

	/// A lower bound on the ratio dist(p,y) / dist(x,y)
	double ymin;

	/// An upper bound on the ratio dist(p,y) / dist(x,y)
	double ymax;
};

/// Manages distance intervals from a set of constraints.
struct DistIntervals {

	/// Initialize to NaN matrices.
	DistIntervals(size_t nObj);

	/// Set the specified distance and interval width
	void set(Eigen::Index ii, Eigen::Index jj, double dmin, double dmax);

	/// Set the appropriate entries from a constraint, if the reference distance
	/// is already set. Returns true if successful.
	bool setFromCon(const TriangleConstraint& con);

	/// The center of each interval. Can be used as distance estimates.
	/// Uninitialized elements have the value NaN.
	Eigen::MatrixXd D;

	/// The width of each interval.
	/// Uninitialized elements have the value NaN.
	Eigen::MatrixXd W;
};

/// Stream a constraint to output
std::ostream& operator<<(std::ostream& os, const TriangleConstraint& con);

/// Sorts constraints into the order needed for embedTrianglesWithSpheres().
/// Throws std::invalid_argument on failure.
void sortTriangleConstraints(std::vector<TriangleConstraint>& cons, bool prune);

/// Validate the triangle constraints to determine whether they fully specify
/// an embedding in some dimensionality.
bool validateTriangleConstraints(const std::vector<TriangleConstraint>& cons,
	nlohmann::json* json);

/// The algorithm to use when selecting reference edges for triangle constraints
enum EdgeMethod {

	/// Pair farthest edges together
	EdgeFarthest,

	/// Select edges by density
	EdgeByDensity,

	/// Select all edges
	EdgeByAll,

	/// Recurse into local regions based on density rules
	EdgeByLocality
};

/// Find triangular constraints for a collection, using some traversal order to
/// identify reference edges. The first point on an edge will be selected by the
/// order, and the second point will be the farthest point from it.
/// If visitBoth = true, the second point will be visited for each edge.
/// This is useful for Collection::frftTraversal() and
/// Collection::stopWhenAffDep(), but not for Collection::stopAfter().
std::vector<TriangleConstraint> findTriangleConstraints(
	std::shared_ptr<Collection> coll, Traversal order, size_t nDim,
	bool visitBoth, bool basicIntervals, EdgeMethod edges);

/// Find triangular constraints for a collection with reference to some pair.
std::vector<TriangleConstraint> findTriangleConstraints(
	std::shared_ptr<Collection> coll, size_t pt1, size_t pt2,
	bool basicIntervals);

/// Find triangular constraints using the specified distance matrix.
/// The constraints will be the correct distances plus/minus eps.
/// Useful for testing.
std::vector<TriangleConstraint> findTriangleConstraints(
	const Eigen::MatrixXd& dists, double eps, size_t nRef);

/// Update triangle constraints using distances from an embedding of a subset
/// of the objects.
std::vector<TriangleConstraint> updateTriangleConstraints(
	const std::vector<TriangleConstraint>& inCons,
	std::shared_ptr<Oracle> ranker, const Eigen::MatrixXd& pos,
	const Eigen::VectorXd& objects);

/// Load triangle constraints from a data file.
std::vector<TriangleConstraint> loadTriangleConstraints(std::string path);

/// Count the number of violated triangle constraints.
size_t numBadTriangles(const std::vector<TriangleConstraint>& cons,
	const Eigen::MatrixXd& dists);

/// Calculate the loss for a single constraint
double triangleLoss(const TriangleConstraint& con,
	const Eigen::MatrixXd& dists);

/// Calculate the loss for a set of triangle constraints and distances.
template<class Container>
double triangleLoss(const Container& cons, const Eigen::MatrixXd& dists) {
	double loss = 0;
	for (const TriangleConstraint& con : cons) {
		loss += triangleLoss(con, dists);
	}
	return loss;
}

/// Calculate the loss for a set of triangle constraints and distances.
template<class ForwardIterator>
double triangleLoss(ForwardIterator begin, ForwardIterator end,
	const Eigen::MatrixXd& dists) {
	double loss = 0;
	for (auto it = begin; it != end; it++) {
		loss += triangleLoss(*it, dists);
	}
	return loss;
}

/// Sorts constraints into the order needed for embedTrianglesWithSpheres()
/// and embedTrianglesWithShells().
/// Throws std::invalid_argument on failure.
void sortConstraints(std::vector<TriangleConstraint>& cons, bool addRotations,
	double maxRotationRatio);

/// Reduce constraint bounds based on embedded coordinates.
/// Used for embedTrianglesWithSpheres() and embedTrianglesWithShells().
std::vector<TriangleConstraint> tightenConstraints(
	const std::map<size_t,std::vector<const TriangleConstraint*>> conMap,
	const Eigen::MatrixXd& X);

/// Find an embedding in R^nDim consistent with the triangle ratio constraints.
/// This method may throw EmbedErr on some errors.
///
/// This method works by first positioning the reference distances, and then
/// placing all the other points with respect to them.
/// It therefore requires that there be an ordering of reference distances such
/// that the points in each distance are constrained by the prior distances.
/// The points of the first reference distance are placed, arbitrarily, at the
/// origin (0, 0, 0, 0, ...) and the first elementary vector (1, 0, 0, 0, ...).
/// The method will attempt to find an ordering of constraints to satisfy its
/// needs, and will throw std::invalid_argument when it can't.
/// The constraints returned by findTriangleConstraints() when it is passed some
/// traversal order are generally valid input.
EmbedResult embedTrianglesWithSpheres(
	std::vector<TriangleConstraint> cons, EmbedConfig config);

/// Find an embedding in R^nDim consistent with the triangle ratio constraints.
/// This method may throw EmbedErr on some errors.
///
/// This method works similarly to embedTrianglesWithSpheres(), but takes the
/// spherical shells into account instead of simply aiming for the center of
/// each shell.
EmbedResult embedTrianglesWithShells(
	std::vector<TriangleConstraint> cons, EmbedConfig config);

/// Find an embedding in R^d (d = X0.cols()) consistent with the triangle
/// ratio constraints.
/// This approach uses a quasi-newton method (L-BFGS) to minimize a loss
/// function defined in terms of the constraints.
/// Returns the embedding and the loss achieved.
EmbedResult embedTrianglesWithOpt(
	std::vector<TriangleConstraint> cons, const Eigen::MatrixXd& X0,
	EmbedConfig config);

/// Find an embedding in R^d consistent with the triangle ratio constraints.
/// Distances are found to a set of d+1 reference vertices using the triangle
/// constraints, and then the set is embedded from the distances.
EmbedResult embedTrianglesWithCM(std::vector<TriangleConstraint> cons,
	EmbedConfig config);

/// Embeds a dataset using a mixture of triangles and comparisons.
EmbedResult embedTrianglesWithMixture(std::vector<TriangleConstraint> tris,
	std::vector<CmpConstraint> cmps, double lambda, const Eigen::MatrixXd& X0,
	EmbedConfig config);

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_TRIANGLE_HPP */
