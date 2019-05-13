/// @file  core.cpp
/// @brief Implements geometric primitives.

#include <ogt/core/core.hpp>
#include <ogt/core/collection.hpp>
#include <algorithm>
#include <map>

using std::any_of;
using std::find;
using std::map;
using std::shared_ptr;
using std::tuple;
using std::vector;

namespace OGT_NAMESPACE {
namespace core {

/// Get points whose distances to each corner is less than the closest other
/// corner.
/// The result is sorted by increasing distance from the first corner.
/// The collection must be sorted by the corners or an UnsortedErr will be
/// thrown.
vector<size_t> lens(shared_ptr<Collection> coll, vector<size_t> corners) {
	if (corners.empty()) {
		return {};
	}

	// Find the min rank of any corner for each corner
	vector<size_t> minRank(corners.size(), coll->nObjGlobal());
	for (size_t i = 0; i < corners.size(); i++) {
		for (size_t j = 0; j < corners.size(); j++) {
			if (i != j) {
				size_t rank = coll->objRank(corners[i], corners[j]);
				if (rank < minRank[i]) {
					minRank[i] = rank;
				}
			}
		}
	}

	// Now find the lens
	vector<size_t> result;
	for (size_t rank = 0; rank <= minRank[0]; rank++) {
		size_t obj = coll->objAt(corners[0], rank);
		bool isBelow = true;
		for (size_t i = 1; i < corners.size(); i++) {
			if (coll->objRank(corners[i], obj) > minRank[i]) {
				isBelow = false;
				break;
			}
		}
		if (isBelow) {
			result.push_back(obj);
		}
	}
	return result;
}

/// Get points whose distances to each corner is less than the distance to apex.
/// The result is sorted by increasing distance from the first corner.
/// The collection must be sorted by the corners or an UnsortedErr will be
/// thrown.
vector<size_t> lens(shared_ptr<Collection> coll, vector<size_t> corners,
	size_t apex) {
	if (corners.empty()) {
		return {};
	}
	vector<size_t> result;
	const size_t maxRank = coll->objRank(corners[0], apex);
	for (size_t rank = 0; rank <= maxRank; rank++) {
		size_t obj = coll->objAt(corners[0], rank);
		bool isBelow = true;
		for (const auto& corner : corners) {
			if (coll->objRank(corner, obj) > coll->objRank(corner, apex)) {
				isBelow = false;
				break;
			}
		}
		if (isBelow) {
			result.push_back(obj);
		}
	}
	return result;
}

/// Get points whose distances to pt1 and pt2 are at most dist(pt1,pt2).
vector<size_t> lens(shared_ptr<Collection> coll, size_t pt1, size_t pt2) {
	return lens(coll, {pt1, pt2});
}

/// Get points closer to pt1 and to pt2 than is the apex point.
vector<size_t> lens(shared_ptr<Collection> coll, size_t pt1, size_t pt2,
	size_t apex) {
	return lens(coll, {pt1, pt2}, apex);
}

/// A helper for countLayersAbove.
vector<size_t> countLayersAboveHelper(shared_ptr<Collection> coll,
	const vector<size_t>& corners, vector<size_t> points,
	map<size_t,size_t>& cache) {
	if (points.empty()) {
		return {};
	} else if (corners.empty()) {
		return vector<size_t>(points.size(), 0);
	}
	const size_t MAX = std::numeric_limits<size_t>::max();
	vector<size_t> count(points.size(), MAX);
	for (size_t ii = 0; ii < points.size(); ii++) {
		auto it = cache.find(points[ii]);
		if (it != cache.end()) {
			count[ii] = it->second;
		} else {

			// Find the corner which gives the maximum rank to the point
			size_t maxRank = 0;
			size_t maxRankCorner = 0;
			for (auto corner : corners) {
				size_t rank = coll->objRank(corner, points[ii]);
				if (rank > maxRank) {
					maxRank = rank;
					maxRankCorner = corner;
				}
			}

			// Now find points above the point starting from the next rank up
			vector<size_t> ptsAbove;
			for (size_t rank = maxRank + 1; rank < coll->nObjLocal(); rank++) {
				size_t obj = coll->objAt(maxRankCorner, rank);
				bool isAbove = true;
				for (auto corner : corners) {
					if (coll->objRank(corner, obj)
						< coll->objRank(corner, points[ii])) {
						isAbove = false;
						break;
					}
				}
				if (isAbove) {
					ptsAbove.push_back(obj);
				}
			}

			// Find out the layer depth of those points, and then find ours.
			if (ptsAbove.empty()) {
				count[ii] = 0;
			} else {
				size_t maxCount = 0;
				for (size_t count : countLayersAboveHelper(coll, corners,
					ptsAbove, cache)) {
					if (count > maxCount) {
						maxCount = count;
					}
				}
				count[ii] = 1 + maxCount;
			}
			cache[points[ii]] = count[ii];
		}
	}
	return count;
}

/// Count the number of "layers" above the given point.
vector<size_t> countLayersAbove(shared_ptr<Collection> coll,
	const vector<size_t>& corners, vector<size_t> points) {
	map<size_t,size_t> cache;
	return countLayersAboveHelper(coll, corners, points, cache);
}

/// Count objects which are farther from all corners than the given point.
size_t countNumAbove(shared_ptr<Collection> coll,
	const vector<size_t>& corners, size_t point) {
	if (corners.empty()) {
		return 0;
	}

	// Start counting from the corner which gives the maximum rank to point
	size_t maxRank = 0;
	size_t maxRankCorner = 0;
	for (auto corner : corners) {
		size_t rank = coll->objRank(corner, point);
		if (rank > maxRank) {
			maxRank = rank;
			maxRankCorner = corner;
		}
	}

	// Now count points starting from the next rank up
	size_t count = 0;
	for (size_t rank = maxRank + 1; rank < coll->nObjLocal(); rank++) {
		size_t obj = coll->objAt(maxRankCorner, rank);
		bool isAbove = true;
		for (auto corner : corners) {
			if (coll->objRank(corner, obj) < coll->objRank(corner, point)) {
				isAbove = false;
				break;
			}
		}
		if (isAbove) {
			count++;
		}
	}
	return count;
}

/// Count objects which are closer to all corners than the given point.
size_t countNumBelow(shared_ptr<Collection> coll, const vector<size_t>& corners,
	size_t point) {
	if (corners.empty()) {
		return 0;
	}

	// Find the min rank corner
	size_t minRank = coll->nObjGlobal();
	size_t minRankCorner = minRank;
	for (size_t corner : corners) {
		size_t rank = coll->objRank(corner, point);
		if (rank < minRank) {
			minRank = rank;
			minRankCorner = corner;
		}
	}

	// Now examine just those objects closer to the min rank corner
	size_t count = 0;
	for (int rank = minRank - 1; rank >= 0; rank--) {
		size_t other = coll->objAt(minRankCorner, rank);
		bool isBelow = true;
		for (const auto& corner : corners) {
			if (coll->objRank(corner, other) >= coll->objRank(corner, point)) {
				isBelow = false;
				break;
			}
		}
		if (isBelow) {
			count++;
		}
	}
	return count;
}

/// Ask whether a point is in or near the convex hull of some set of "corner"
/// points.
/// The collection must be sorted by the corner points, or UnsortedErr will be
/// thrown.
/// If the collection is also sorted by the query point, a more precise
/// answer will be given.
bool isNearConv(shared_ptr<Collection> coll, const vector<size_t>& corners,
	size_t query) {

	// Trivial base cases
	if (corners.empty()) {
		return false;
	} else if (corners.size() == 1) {
		return corners[0] == query;
	}

	// Corner test: query is not in conv if there exists another point which is
	// closer to all corners than is query.
	size_t minRank = coll->nObjGlobal();
	size_t minRankCorner = 0;
	for (const auto& corner : corners) {
		size_t rank = coll->objRank(corner, query);
		if (rank < minRank) {
			minRank = rank;
			minRankCorner = corner;
		}
	}
	for (size_t rank = 0; rank < minRank; rank++) {
		size_t obj = coll->objAt(minRankCorner, rank);
		bool closerToAll = true;
		for (const auto& corner : corners) {
			if (coll->objRank(corner, obj) > coll->objRank(corner, query)) {
				closerToAll = false;
				break;
			}
		}
		if (closerToAll) {
			return false;
		}
	}

	// Query point test: query is not in conv if it disagrees on the ordering of
	// any pair for which the corners all agree.
	if (coll->isSortedBy(query)) {
		// TODO: implement query point test for isNearConv(), e.g. using
		// concordant/discordant count like n log n Kendall's tau calculation
	}

	// Guess that query is in conv if we have not ruled it out
	return true;
}

/// A helper for conv. Removes too-distant points from in_conv.
void nearConvHelper(shared_ptr<Collection> coll, size_t query,
	const vector<size_t>& objects, vector<bool>& in_conv,
	const vector<size_t>& corners) {

	vector<bool> in_query_conv(in_conv.size(), false);
	for (auto corner : corners) {
		const size_t qr = coll->objRank(corner, query);
		for (size_t oi = 0; oi < in_query_conv.size(); ++oi) {
			if (coll->objRank(corner, objects[oi]) <= qr) {
				in_query_conv[oi] = true;
			}
		}
	}
	for (size_t oi = 0; oi < in_query_conv.size(); oi++) {
		if (!in_query_conv[oi]) {
			in_conv[oi] = false;
		}
	}
}

/// Find points in or near the convex hull of some set of points.
vector<size_t> nearConv(shared_ptr<Collection> coll,
	const vector<size_t>& corners) {

	// Handle base cases
	if (corners.empty()) {
		return vector<size_t>();
	} else if (corners.size() == 1) {
		return corners;
	}

	// Sort by all the corners
	for (size_t corner : corners) {
		if (!coll->isSortedBy(corner)) {
			coll->sort(corner);
		}
	}

	// Start by using a corner as a query point to reduce the number of calls
	// we make to nearConvHelper.
	const auto& objects = coll->objects();
	vector<bool> in_conv(objects.size(), true);
	nearConvHelper(coll, corners[0], objects, in_conv, corners);

	// Continue using every point still in in_conv as a query point
	for (size_t oi = 0; oi < objects.size(); oi++) {
		const size_t query = objects[oi];
		if (query != corners[0] && in_conv[oi]) {
			nearConvHelper(coll, query, objects, in_conv, corners);
		}
	}

	// Build the output.
	vector<size_t> result;
	for (size_t oi = 0; oi < objects.size(); oi++) {
		if (in_conv[oi]) {
			result.push_back(objects[oi]);
		}
	}
	sort(result.begin(), result.end(), [&](size_t a, size_t b) {
		return coll->objRank(corners[0], a) < coll->objRank(corners[0], b);
	});
	return result;
}

/// Identify points on or near the line between two points in the collection.
vector<size_t> nearLine(shared_ptr<Collection> coll, size_t pt1, size_t pt2) {
	return nearConv(coll, {pt1, pt2});
}

/// Use binary search to find a line midpoint
size_t findLineMidpointHelper(std::shared_ptr<Collection> coll,
	const vector<size_t>& line, size_t lMid, size_t rMid) {
	// Invariant: all points from lMid to left are closer to left, and all
	// points from rMid to right are closer to right.

	// Base cases
	if (lMid >= rMid - 1) {
		return lMid;
	}

	size_t mid = (lMid + rMid) / 2;
	switch(coll->sortPair(line[mid], line.front(), line.back())) {
		case AB_LT_AC:
			// mid closer to left, so check to its "right"
			return findLineMidpointHelper(coll, line, mid, rMid);
		case AB_EQ_AC:
			// Found it!
			return mid;
		case AC_LT_AB:
			// mid closer to right, so check to its "left"
			return findLineMidpointHelper(coll, line, lMid, mid);
		default: // unreachable in practice?
			return mid;
	}
}

/// Identify the midpoint of a line segment.
size_t findLineMidpoint(std::shared_ptr<Collection> coll,
	const std::vector<size_t>& line) {
	if (line.size() <= 2) {
		return 0;
	}
	return findLineMidpointHelper(coll, line, 0, line.size() - 1);
}

/// Ask whether a point is affinely indepenent of some set of "corner" points.
bool isAffinelyIndependent(shared_ptr<Collection> coll, size_t point,
	const vector<size_t>& corners) {

	// Base cases
	if (corners.empty()) {
		return true;
	} else if (corners.size() == 1) {
		return point != corners[0];
	} else if (find(corners.begin(), corners.end(), point) != corners.end()) {
		return false;
	}

	// Find points in the convex hull of the set of all points
	vector<size_t> all(corners);
	all.push_back(point);
	vector<bool> isNew(coll->nObjGlobal(), false);
	for (size_t obj : nearConv(coll, all)) {
		isNew[obj] = true;
	}

	// Eliminate points which are in the convex hulls of subsets of endpoints
	for (size_t obj : nearConv(coll, corners)) {
		isNew[obj] = false;
	}
	for (size_t idx = 0; idx < corners.size(); idx++) {
		vector<size_t> rest(corners);
		rest[idx] = point;
		for (size_t obj : nearConv(coll, rest)) {
			isNew[obj] = false;
		}
	}

	// The points are affinely independent if there is any remaining point
	return any_of(isNew.begin(), isNew.end(), [](bool isn) { return isn; });
}

} // end namespace core
} // end namespace OGT_NAMESPACE
