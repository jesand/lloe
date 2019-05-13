/// @file  core.hpp
/// @brief Geometric primitives used to build more complex algorithms.

#pragma once
#ifndef OGT_CORE_CORE_HPP
#define OGT_CORE_CORE_HPP

#include <ogt/config.hpp>
#include <memory>
#include <tuple>
#include <vector>

namespace OGT_NAMESPACE {
namespace core {

struct Collection; // Forward declaration

/// Get points whose distances to each corner is less than the closest other
/// corner.
/// The result is sorted by increasing distance from the first corner.
/// The collection must be sorted by the corners or an UnsortedErr will be
/// thrown.
std::vector<size_t> lens(std::shared_ptr<Collection> coll,
	std::vector<size_t> corners);

/// Get points whose distances to each corner is less than the distance to apex.
/// The result is sorted by increasing distance from the first corner.
/// The collection must be sorted by the corners or an UnsortedErr will be
/// thrown.
std::vector<size_t> lens(std::shared_ptr<Collection> coll,
	std::vector<size_t> corners, size_t apex);

/// Get points whose distances to pt1 and pt2 are at most dist(pt1,pt2).
/// The result is sorted by increasing distance from pt1.
/// The collection must be sorted by pt1 and pt2 or an UnsortedErr will be
/// thrown.
std::vector<size_t> lens(std::shared_ptr<Collection> coll, size_t pt1,
	size_t pt2);

/// Get points closer to pt1 and to pt2 than is the apex point.
/// The result is sorted by increasing distance from pt1.
/// The collection must be sorted by pt1 and pt2 or an UnsortedErr will be
/// thrown.
std::vector<size_t> lens(std::shared_ptr<Collection> coll, size_t pt1,
	size_t pt2, size_t apex);

/// Count the number of "layers" above the given point. That is, find the length
/// of the longest sequence p_1, ..., p_k where p_1 is point and each p_i is
/// contained in lens(corners, p_{i+1}).
/// The collection must already be sorted by the corners, or an UnsortedErr
/// will be thrown.
std::vector<size_t> countLayersAbove(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& corners, std::vector<size_t> points);

/// Count objects which are farther from all corners than the given point.
/// The collection must already be sorted by the corners, or an UnsortedErr
/// will be thrown.
size_t countNumAbove(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& corners, size_t point);

/// Count objects which are closer to all corners than the given point.
/// This is one less than the number of points in lens(coll, corners, point).
/// If the count is nonzero, then the point cannot be in the convex hull of the
/// corners, and all points summed up in the count are either closer to the hull
/// or inside it.
/// The collection must already be sorted by the corners, or an UnsortedErr
/// will be thrown.
size_t countNumBelow(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& corners, size_t point);

/// Ask whether a point is in or near the convex hull of some set of "corner"
/// points.
/// The collection must be sorted by the corner points, or UnsortedErr will be
/// thrown.
/// If the collection is also sorted by the query point, a more precise
/// answer will be given.
bool isNearConv(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& corners, size_t query);

/// Find points in or near the convex hull of some set of "corner" points.
/// If the collection has not been sorted with respect to the corners, then
/// this will be done first.
/// The output will be presented in order of increasing distance from the first
/// corner.
///
/// This method is designed to minimize the number of sort() calls used, at the
/// expense of including some "false positive" points which could otherwise be
/// excluded from the result.
/// We identify points near the convex hull as follows.
/// For any query point q, the set containing q and all points which are
/// closer to some anchor than q is will contain the convex hull and,
/// typically, many extra points.
/// We prune out as many extra points as possible by taking the intersection
/// of this set over many query points.
std::vector<size_t> nearConv(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& corners);

/// Identify points on or near the line segment between two points in the
/// collection. The results are sorted by distance to pt1.
std::vector<size_t> nearLine(std::shared_ptr<Collection> coll,
	size_t pt1, size_t pt2);

/// Identify the midpoint of a line segment. Given the output from nearLine(),
/// return the index of the last point in the vector whose projection onto the
/// line is before the midpoint.
size_t findLineMidpoint(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& line);

/// Ask whether a point is affinely indepenent of some set of "corner" points.
/// This test operates by repeated calls to conv().
bool isAffinelyIndependent(std::shared_ptr<Collection> coll, size_t point,
	const std::vector<size_t>& corners);

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_CORE_HPP */
