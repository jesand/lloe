/// @file  collection.hpp
/// @brief Declarations related to object collections.

#pragma once
#ifndef OGT_CORE_COLLECTION_HPP
#define OGT_CORE_COLLECTION_HPP

#include <ogt/config.hpp>
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <ogt/core/oracle.hpp>
#include <set>
#include <string>
#include <vector>

namespace OGT_NAMESPACE {

// Forward declaration
namespace embed {
	struct CmpConstraint;
}

namespace core {

/// The type thrown on I/O errors.
struct UnsortedErr : public std::runtime_error {

	/// Build an UnsortedErr with the specified message.
	UnsortedErr(std::string message)
		: std::runtime_error("ogt::UnsortedErr: " + message) {
	}
};

class Traversal; // Forward declaration

/// The collection of objects/entities you are analyzing.
/// This is a container of unique identifiers for objects paired with a Sorter
/// which defines the object ordering, along with any ordinal data inferred so
/// far about the objects.
struct Collection : std::enable_shared_from_this<Collection> {

	/// Construct a collection with object IDs 0 to nObj-1.
	static std::shared_ptr<Collection> Create(std::shared_ptr<Sorter> sorter,
		size_t nObj);

	/// Shorthand for creating a collection with an OracleSorter.
	static std::shared_ptr<Collection> Create(std::shared_ptr<Oracle> oracle,
		size_t nObj);

	/// Construct a collection with a subset of IDs from some global collection.
	static std::shared_ptr<Collection> Create(std::shared_ptr<Oracle> oracle,
		size_t nObj, std::vector<size_t> objects);

	/// Construct a collection with a subset of IDs from some global collection.
	static std::shared_ptr<Collection> Create(std::shared_ptr<Sorter> sorter,
		size_t nObj, std::vector<size_t> objects);

	/// Construct a collection containing a subset of this one
	virtual std::shared_ptr<Collection> subset(std::vector<size_t> items)
		const = 0;

	/// Virtual destructor.
	virtual ~Collection() = default;

	/// Get the number of objects in the global collection.
	virtual size_t nObjGlobal() const = 0;

	/// Get the number of objects in the local collection.
	virtual size_t nObjLocal() const = 0;

	/// Get the ID of the nth object in the collection.
	virtual size_t object(size_t n) const = 0;

	/// Get all object IDs in the collection.
	virtual const std::vector<size_t>& objects() const = 0;

	/// Get a random object from the collection.
	virtual size_t randObject() const = 0;

	/// Sort a pair of objects by distance to `a`. Avoids invoking the sorter
	/// when possible.
	virtual Cmp sortPair(size_t a, size_t b, size_t c) = 0;

	/// Sort the collection by distance from a head object.
	/// Stores the results for future reference via objRank() and objAt().
	virtual void sort(size_t head) = 0;

	/// Sort a vector of object indices in-place by distance to the head object.
	/// The collection must have been sorted by the head already, or
	/// UnsortedErr will be thrown.
	virtual void sort(size_t head, std::vector<size_t>& objects) = 0;

	/// Sort the collection by distance to the head object, and place the
	/// corresponding comparisons into the provided vector.
	virtual void sort(size_t head, std::vector<embed::CmpConstraint>& cmps) = 0;

	/// Ask whether the collection has been sorted for a particular head yet.
	virtual bool isSortedBy(size_t head) const = 0;

	/// Get the rank of an object from a given head or throw UnsortedErr.
	virtual size_t objRank(size_t head, size_t object) const = 0;

	/// Get the ranks of all objects from a given head or throw UnsortedErr.
	virtual std::vector<size_t> objRanks(size_t head) const = 0;

	/// Get the object at a given rank from a given head or throw UnsortedErr.
	virtual size_t objAt(size_t head, size_t rank) const = 0;

	/// Get the order of all objects from a given head or throw UnsortedErr.
	virtual std::vector<size_t> objOrder(size_t head) const = 0;

	/// Get the objects with ranks `rank` and below. If the object is sorted,
	/// the results will be sorted by increasing distance to the head. If not,
	/// they will appear in arbitrary order.
	/// If the object is not sorted, this takes O(nObjLocal()) comparisons.
	/// @see isSorted()
	virtual std::vector<size_t> objKnn(size_t head, size_t rank) = 0;

	/// Get the object at maximum rank from a given head.
	/// This object is guaranteed to be on the convex hull of the (local)
	/// object collection, assuming the order is correct.
	virtual size_t farthestFrom(size_t head) = 0;

	/// Estimate the dimensionality of the set.
	/// This selects a subsample of at most maxObj objects, and does a FRFT
	/// traversal until the set is affinely independent. It then returns the
	/// implied dimensionality.
	/// Increasing maxObj can lead to a better estimate, as the estimation
	/// process is more accurate with higher density subsets. However, it also
	/// costs more, both in terms of comparisons used and CPU cycles.
	virtual size_t estimateDim(size_t maxObj) const = 0;

	/// Find a subset of points which is evenly distributed through the
	/// collection.
	virtual std::set<size_t> findEvenSubset(const std::set<size_t>& refs,
		size_t nPts) = 0;

	/// Traverse in ascending ID order.
	/// @see Traversal::AscId(), AscIdNexter
	virtual Traversal ascIdTraversal() = 0;

	/// Traverse in random order.
	/// @see Traversal::Random(), RandomNexter
	virtual Traversal randomTraversal() = 0;

	/// Traverse in FRFT order. The points traversed need to be sorted as we go,
	/// so this cannot be done with a const Collection.
	/// @see Traversal::FRFT(), FrftNexter
	virtual Traversal frftTraversal() = 0;

	/// Traverse all points by distance to the head. Sorts by head.
	/// @see Traversal::Rank(), RankNexter
	virtual Traversal rankTraversal(size_t head) = 0;

	/// Traverse the points in a lens by distance to pt1. Sorts by pt1 and pt2.
	/// @see Traversal::Lens(), LensNexter
	virtual Traversal lensTraversal(size_t pt1, size_t pt2) = 0;

	/// Traverse the points in Above Max order.
	/// @see Traversal::AboveMax(), AboveMaxNexter
	virtual Traversal aboveMaxTraversal() = 0;

	/// Split into random subsets.
	/// @see Splitter::Random, RandomSplitter.
	virtual std::vector<std::shared_ptr<Collection>> splitRandom(
		size_t nOverlap) = 0;

	/// Split into subsets of the kNN of visited objects.
	/// @see Splitter::Random, RandomSplitter.
	virtual std::vector<std::shared_ptr<Collection>> splitTraversalKnn(
		Traversal order) = 0;

	/// Saves random true triples to the specified sink.
	virtual void saveRandomTriples(std::ostream& out, size_t numTriples) = 0;

	/// Returns the set of points within the ball centered on `center` and with
	/// radius `dist(center, radius)`.
	/// The parameter closed indicates whether the ball is open (excluding
	/// radius) or closed (including radius).
	/// If subset is not None, only points from subset which lie in the ball
	/// are returned.
	virtual std::set<size_t> ball(size_t center, size_t radius,
		bool closed=true, const std::set<size_t>& subset=std::set<size_t>()) = 0;

	/// Determines whether point(s) `x` lie within the ball centered on `center`
	/// and with radius `dist(center, radius)`.
	virtual bool in_ball(size_t x, size_t center, size_t radius,
		bool closed=true) = 0;

	/// Returns the number of points in lens(p, q).
	virtual size_t lens_size(size_t p, size_t q, bool closed=true) = 0;

	/// Return a set containing the points x for which edge pq is the longest in
	/// triangle xpq.
	virtual std::set<size_t> lens(size_t p, size_t q, bool closed=true, 
		const std::set<size_t>& subset=std::set<size_t>()) = 0;

	/// Asks whether x is in lens(p, q).
	virtual bool in_lens(size_t x, size_t p, size_t q, bool closed=true) = 0;
};

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_COLLECTION_HPP */
