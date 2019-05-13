/// @file  traversal.hpp
/// @brief Declarations to permit collection traversal in various orders

#pragma once
#ifndef OGT_CORE_TRAVERSAL_HPP
#define OGT_CORE_TRAVERSAL_HPP

#include <ogt/config.hpp>
#include <functional>
#include <memory>
#include <vector>

namespace OGT_NAMESPACE {
namespace core {

struct Collection; // Forward declaration

/// An abstract rule for terminating a Traversal early.
struct Stopper {

	/// Virtual destructor
	virtual ~Stopper() = default;

	/// Decide whether to stop based on having visited the specified items.
	virtual bool shouldStop(const std::vector<size_t>& visited, size_t num) = 0;
};

/// An abstract class used to define a concrete traversal order.
class Nexter {
public:

	/// Virtual destructor
	virtual ~Nexter() = default;

	/// Set the stopper pointer.
	void setStopper(std::shared_ptr<Stopper> stopper) {
		this->stopper = stopper;
	}

	/// Return the position to use for the end() iterator
	virtual size_t endPos() const = 0;

	/// Return the item at the specified index, generating the order as needed.
	virtual size_t& at(size_t pos) = 0;

	/// Insert an item into the specified position in traversal order, but only
	/// if it was not already at some prior position. Returns true if the order
	/// of traversal was changed.
	virtual bool visit(size_t pos, size_t obj) = 0;

	/// Ask whether we should stop at the given position.
	virtual bool shouldStopAt(size_t pos) = 0;

	/// Clone the Nexter instance for the specified collection.
	virtual std::shared_ptr<Nexter> forColl(std::shared_ptr<Collection> coll) = 0;

protected:

	/// The stopping rule to use for iterators.
	std::shared_ptr<Stopper> stopper;
};

/// A traversal of the collection in some specified order
class Traversal {
public:

	/// Create a FRFT traversal of a collection. Sorts by each visited point, so
	/// a full traversal requires Theta(n^2*log(n)) comparisons.
	/// @see Collection::frftTraversal(), FrftNexter
	static Traversal FRFT(std::shared_ptr<Collection> coll);

	/// Create a random traversal of a collection.
	/// @see Collection::randomTraversal(), RandomNexter
	static Traversal Random(std::shared_ptr<Collection> coll);

	/// Create a traversal in increasing lexicographic/numeric order.
	/// @see Collection::ascIdTraversal(), AscIdNexter
	static Traversal AscId(std::shared_ptr<Collection> coll);

	/// Create a traversal in order of ascending distance from the specified
	/// point. Sorts by head if necessary.
	/// @see Collection::rankTraversal(), RankNexter
	static Traversal Rank(std::shared_ptr<Collection> coll, size_t head);

	/// Create a traversal of the contents of a lens in distance order from pt1.
	/// Sorts by pt1 and pt2 if necessary.
	/// @see Collection::lensTraversal(), LensNexter
	static Traversal Lens(std::shared_ptr<Collection> coll, size_t pt1, size_t pt2);

	/// Create a traversal which selects the point above the most points, with
	/// respect to the convex hull of previously-visited points.
	/// This will not visit the entire collection. Instead, it will visit points
	/// while unexplored dimensions still exist.
	/// @see Collection:aboveMaxTraversal(), AboveMaxNexter
	static Traversal AboveMax(std::shared_ptr<Collection> coll);

	/// Apply the traversal to the specified collection.
	Traversal forColl(std::shared_ptr<Collection> coll);

	/// Stop after some maximum number of items have been visited.
	/// @see addStopper(), MaxCountStopper.
	Traversal stopAfter(size_t maxCount);

	/// Stop if the visited items are not affinely independent.
	/// @see addStopper(), AffDepStopper.
	Traversal stopWhenAffDep();

	/// Build an instance with the specified nexter
	Traversal(std::shared_ptr<Collection> coll, std::shared_ptr<Nexter> nexter);

	/// Copy constructor
	Traversal(const Traversal& copy);

	/// A forward iterator in some traversal order.
	//  Warning: Implementing a random-access iterator is a little tricky,
	//    due to the visit() implementations used.
	class iterator : std::iterator<std::forward_iterator_tag, size_t> {
	public:

		/// Constructor.
		iterator(size_t pos, std::shared_ptr<Nexter> nexter);

		/// Copy constructor.
		iterator(const iterator& copy);

		/// Assignment.
		iterator& operator=(const iterator& copy);

		/// Preincrement.
		iterator& operator++();

		/// Postincrement.
		iterator operator++(int);

		/// Dereference.
		size_t operator*();

		/// Dereference.
		const size_t* operator->();

		/// Inequality.
		bool operator!=(const iterator& copy) const { return !(*this == copy); }

		/// Equality.
		bool operator==(const iterator& copy) const;

		/// Get the current position. This is a helper for Traversal::visit().
		size_t curpos() const { return pos; }

	private:

		/// The current position
		size_t pos;

		/// Emits the next object as needed
		std::weak_ptr<Nexter> nexter;
	};

	/// Get an iterator to the beginning of a traversal.
	iterator begin();

	/// Get an iterator just past the end of a traversal.
	iterator end();

	/// Insert an object into the traversal order just after iter, but only if
	/// it was not already in some position before iter. Returns an iterator at
	/// the new position of obj if its position was changed, and returns iter
	/// otherwise. This is meant to facilitate skipping an object in iteration
	/// order.
	///
	///     // Skip objects following even-numbered objects
	///     auto order = Traversal::AscId(coll);
	///     for (auto iter = order.begin(); iter != order.end(); iter++) {
	///         if (*iter % 2 == 0) {
	///             iter = order.visit(*iter + 1);
	///         }
	///     }
	iterator visit(iterator iter, size_t obj);

	/// Add a stopping rule which can terminate the traversal early.
	/// Behavior is undefined if used mid-traversal.
	///
	///     auto order = coll->randomTraversal();
	///     // This loop might or might not use the stop rule consistently.
	///     for(size_t obj : order) {
	///         order.addStopper(...); // May or may not affect loop termination
	///         std::cout << obj << std::endl;
	///     }
	///     // This loop will use the stop rule.
	///     for(size_t obj : order) {
	///         std::cout << obj << std::endl; // Prints until stopper applies
	///     }
	Traversal addStopper(std::shared_ptr<Stopper> stopper);

private:

	/// The collection we are iterating over
	std::shared_ptr<Collection> coll;

	/// The object used to generate and preserve the traversal order
	std::shared_ptr<Nexter> nexter;
};

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_TRAVERSAL_HPP */
