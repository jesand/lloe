/// @file  traversal.cpp
/// @brief Definitions related to collection traversal.

#include <algorithm>
#include <ogt/core/traversal.hpp>
#include <ogt/core/collection.hpp>
#include <ogt/core/core.hpp>
#include <ogt/util/random.hpp>

using OGT_NAMESPACE::util::globalRand;
using OGT_NAMESPACE::util::random_choice;
using std::shared_ptr;
using std::vector;

namespace OGT_NAMESPACE {
namespace core {

/// Constructor.
Traversal::Traversal(shared_ptr<Collection> coll, shared_ptr<Nexter> nexter)
	: coll(coll)
	, nexter(nexter)
{
}

/// Copy constructor.
Traversal::Traversal(const Traversal& copy) {
	coll = copy.coll;
	nexter = copy.nexter;
}

/// Apply the traversal to the specified collection.
Traversal Traversal::forColl(shared_ptr<Collection> coll) {
	return Traversal(coll, nexter->forColl(coll));
}

/// Constructor.
Traversal::iterator::iterator(size_t pos, shared_ptr<Nexter> nexter)
	: pos(pos), nexter(nexter)
{
}

/// Copy constructor.
Traversal::iterator::iterator(const iterator& copy)
	: pos(copy.pos), nexter(copy.nexter)
{
}

/// Assignment.
Traversal::iterator& Traversal::iterator::operator=(
	const Traversal::iterator& copy) {
	pos = copy.pos;
	nexter = copy.nexter;
	return *this;
}

/// Preincrement.
Traversal::iterator& Traversal::iterator::operator++() {
	pos++;
	return *this;
}

/// Postincrement.
Traversal::iterator Traversal::iterator::operator++(int) {
	iterator old(*this);
	operator++();
	return old;
}

/// Dereference.
size_t Traversal::iterator::operator*() {
	if (auto sh = nexter.lock()) {
		return sh->at(pos);
	}
	throw std::runtime_error(
		"Attempt to dereference Traversal::iterator with deleted Nexter");
}

/// Dereference.
const size_t* Traversal::iterator::operator->() {
	if (auto sh = nexter.lock()) {
		return &sh->at(pos);
	}
	throw std::runtime_error(
		"Attempt to dereference Traversal::iterator with deleted Nexter");
}

/// Equality.
bool Traversal::iterator::operator==(const Traversal::iterator& copy) const {
	auto sh = nexter.lock();
	if (sh == copy.nexter.lock()) {
		if (sh) {
			return (pos == copy.pos)
				|| (sh->shouldStopAt(pos) && sh->shouldStopAt(copy.pos));
		}
		throw std::runtime_error(
			"Attempt to dereference Traversal::iterator with deleted Nexter");
	} else {
		return false;
	}
}

/// Get an iterator to the beginning of a traversal.
Traversal::iterator Traversal::begin() {
	return Traversal::iterator(0, nexter);
}

/// Get an iterator just past the end of a traversal.
Traversal::iterator Traversal::end() {
	return Traversal::iterator(nexter->endPos(), nexter);
}

/// Insert an object into the traversal order just after iter, but only if
/// it was not already in some position before iter.
Traversal::iterator Traversal::visit(Traversal::iterator iter, size_t obj) {
	if (nexter->visit(iter.curpos() + 1, obj)) {
		return Traversal::iterator(iter.curpos() + 1, nexter);
	} else {
		return iter;
	}
}

/// Add a stopping rule which can terminate the traversal early.
Traversal Traversal::addStopper(shared_ptr<Stopper> stopper) {
	this->nexter->setStopper(stopper);
	return *this;
}

/// Generates a distinct farthest-rank-first traversal order
/// @see Collection::frftTraversal(), Traversal::FRFT()
class FrftNexter : public Nexter {
public:

	/// Constructor
	FrftNexter(shared_ptr<Collection> coll)
		: coll(coll)
		, order()
		, pending()
	{
		order.push_back(genFirst());
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		while (pos >= order.size()) {
			order.push_back(genNext());
		}
		return order[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return coll->nObjLocal();
	}

	/// Insert an item into the specified position in traversal order, if new.
	bool visit(size_t pos, size_t obj) override {
		if (find(order.begin(), order.end(), obj) != order.end()) {
			return false;
		}
		if (pos >= order.size()) {
			order.push_back(obj);
		} else {
			order.insert(order.begin() + pos, obj);
		}
		updatePending(obj);
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		if (pos >= endPos()) {
			return true;
		} else {
			at(pos); // generate new items to visit
			return stopper && stopper->shouldStop(order, pos + 1);
		}
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<FrftNexter>(coll);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

	/// Identify the first point in FRFT traversal order
	size_t genFirst() {
		size_t p0 = coll->randObject();
		coll->sort(p0);
		size_t choice = coll->farthestFrom(p0);
		coll->sort(choice);

		pending.resize(coll->nObjLocal());
		for (size_t rank = 1; rank < pending.size(); rank++) {
			pending[rank].push_back(coll->objAt(choice, rank));
		}
		return choice;
	}

	/// Identify the next point in FRFT traversal order
	size_t genNext() {

		// Pick the next point furthest from the last in the max min rank bucket
		vector<size_t>& maxmin = pending.back();
		const size_t last = order.back();
		vector<size_t>::iterator it;
		size_t maxRank = 0;
		for (auto iter = maxmin.begin(); iter != maxmin.end(); iter++) {
			size_t rank = coll->objRank(last, *iter);
			if (rank > maxRank) {
				it = iter;
				maxRank = rank;
			}
		}
		size_t choice = *it;
		maxmin.erase(it);
		updatePending(choice);
		return choice;
	}

	/// Updates the pending buckets
	void updatePending(size_t obj) {
		coll->sort(obj);
		for (size_t r = 1; r < pending.size(); r++) {
			for (auto it = pending[r].begin(); it != pending[r].end(); ) {
				size_t newR = coll->objRank(obj, *it);
				if (newR < r) {
					pending[newR].push_back(*it);
					it = pending[r].erase(it);
				} else {
					it++;
				}
			}
		}
		while (!pending.empty() && pending.back().empty()) {
			pending.pop_back();
		}
	}

private:

	/// The collection we traverse over
	shared_ptr<Collection> coll;

	/// The traversal order so far
	vector<size_t> order;

	/// The pending objects, indexed by min rank
	vector<vector<size_t>> pending;
};

/// Return a new FRFT traversal order
Traversal Traversal::FRFT(shared_ptr<Collection> coll) {
	return Traversal(coll, std::make_shared<FrftNexter>(coll));
}

/// Generates a distinct random traversal order
/// @see Collection::randomTraversal(), Traversal::Random()
class RandomNexter : public Nexter {
public:

	/// Constructor
	RandomNexter(const shared_ptr<Collection> coll)
		: order(coll->objects())
	{
		auto& rng = globalRand();
		std::shuffle(order.begin(), order.end(), rng);
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		return order[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return order.size();
	}

	/// Insert an item into the specified position in traversal order, if its
	/// position was after pos.
	bool visit(size_t pos, size_t obj) override {
		size_t oldPos = find(order.begin(), order.end(), obj) - order.begin();
		if (oldPos < pos) {
			return false;
		}
		order[oldPos] = order[pos];
		order[pos] = obj;
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		return pos >= endPos()
			|| (stopper && stopper->shouldStop(order, pos + 1));
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<RandomNexter>(coll);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

private:

	/// The traversal order so far
	vector<size_t> order;
};

/// Create a random traversal of a collection.
Traversal Traversal::Random(shared_ptr<Collection> coll) {
	return Traversal(coll, std::make_shared<RandomNexter>(coll));
}

/// A traversal order in ascending object ID order.
/// @see Collection::ascIdTraversal(), Traversal::AscId()
class AscIdNexter : public Nexter {
public:

	/// Constructor
	AscIdNexter(const shared_ptr<Collection> coll)
		: order(coll->objects())
	{
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		return order[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return order.size();
	}

	/// Insert an item into the specified position in traversal order, if its
	/// position was after pos.
	bool visit(size_t pos, size_t obj) override {
		auto it = find(order.begin(), order.end(), obj);
		if (static_cast<size_t>(it - order.begin()) < pos) {
			return false;
		}
		order.erase(it);
		order.insert(order.begin() + pos, obj);
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		return pos >= endPos()
			|| (stopper && stopper->shouldStop(order, pos + 1));
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<AscIdNexter>(coll);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

private:

	/// The traversal order so far
	vector<size_t> order;
};

/// Create a traversal in increasing lexicographic/numeric order.
Traversal Traversal::AscId(shared_ptr<Collection> coll) {
	return Traversal(coll, std::make_shared<AscIdNexter>(coll));
}

/// A traversal order in ascending rank order.
/// @see Collection::rankTraversal(), Traversal::Rank()
class RankNexter : public Nexter {
public:

	/// Constructor
	RankNexter(const shared_ptr<Collection> coll, size_t head)
		: head(head)
		, order(coll->objects())
	{
		coll->sort(head);
		std::sort(order.begin(), order.end(), [&](auto a, auto b) {
			return coll->objRank(head, a) < coll->objRank(head, b);
		});
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		return order[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return order.size();
	}

	/// Insert an item into the specified position in traversal order, if its
	/// position was after pos.
	bool visit(size_t pos, size_t obj) override {
		auto it = find(order.begin(), order.end(), obj);
		if (static_cast<size_t>(it - order.begin()) < pos) {
			return false;
		}
		order.erase(it);
		order.insert(order.begin() + pos, obj);
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		return pos >= endPos()
			|| (stopper && stopper->shouldStop(order, pos + 1));
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<RankNexter>(coll, head);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

private:

	/// The head we are sorting by
	const size_t head;

	/// The traversal order so far
	vector<size_t> order;
};

/// Create a traversal in order of ascending distance from the specified
/// point. Sorts by head if necessary.
Traversal Traversal::Rank(std::shared_ptr<Collection> coll, size_t head) {
	return Traversal(coll, std::make_shared<RankNexter>(coll, head));
}

/// A traversal order of the contents of a lens in ascending distance from the
/// first lens endpoint.
/// @see Collection::lensTraversal(), Traversal::Lens()
class LensNexter : public Nexter {
public:

	/// Constructor
	LensNexter(shared_ptr<Collection> coll, size_t pt1, size_t pt2)
		: pt1(pt1)
		, pt2(pt2)
	{
		coll->sort(pt1);
		coll->sort(pt2);
		size_t maxRank2 = coll->objRank(pt2, pt1);
		for (size_t rank = 0; rank <= coll->objRank(pt1, pt2); rank++) {
			size_t obj = coll->objAt(pt1, rank);
			if (coll->objRank(pt2, obj) <= maxRank2) {
				order.push_back(obj);
			}
		}
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		return order[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return order.size();
	}

	/// Insert an item into the specified position in traversal order, if its
	/// position was after pos.
	bool visit(size_t pos, size_t obj) override {
		auto it = find(order.begin(), order.end(), obj);
		if (it != order.end()) {
			size_t oldPos = it - order.begin();
			if (oldPos < pos) {
				return false;
			} else {
				order.erase(it);
			}
		}
		order.insert(order.begin() + pos, obj);
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		return pos >= endPos()
			|| (stopper && stopper->shouldStop(order, pos + 1));
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<LensNexter>(coll, pt1, pt2);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

private:

	/// The first point that defines the lens
	const size_t pt1;

	/// The second point that defines the lens
	const size_t pt2;

	/// The traversal order
	vector<size_t> order;
};

/// Create a traversal of the contents of a lens in distance order from pt1.
Traversal Traversal::Lens(shared_ptr<Collection> coll, size_t pt1, size_t pt2) {
	return Traversal(coll, std::make_shared<LensNexter>(coll, pt1, pt2));
}

/// Visits points above the maximum number of points w.r.t. the convex hull of
/// the previously-visited points.
/// @see Collection::aboveMaxTraversal(), Traversal::AboveMax()
class AboveMaxNexter : public Nexter {
public:

	/// Constructor
	AboveMaxNexter(shared_ptr<Collection> coll)
		: coll(coll)
		, pending(coll->objects())
	{
		size_t p0 = coll->randObject();
		coll->sort(p0);
		updateOrder(0, coll->farthestFrom(p0));
	}

	/// Return the item at the specified index, generating the order as needed.
	size_t& at(size_t pos) override {
		while (pos >= corners.size() && !active.empty()) {
			auto it = random_choice(active.begin(), active.end());
			size_t choice = *it;
			active.erase(it);
			updateOrder(corners.size(), choice);
		}
		if (pos >= corners.size()) {
			throw std::runtime_error(
				"Attempted to access an invalid position in AboveMaxNexter");
		}
		return corners[pos];
	}

	/// Return the position to use for the end() iterator
	size_t endPos() const override {
		return coll->nObjLocal();
	}

	/// Insert an item into the specified position in traversal order, if new.
	bool visit(size_t pos, size_t obj) override {
		if (find(corners.begin(), corners.end(), obj) != corners.end()) {
			return false;
		}
		updateOrder(pos, obj);
		return true;
	}

	/// Ask whether we should stop at the given position.
	bool shouldStopAt(size_t pos) override {
		if (pos >= endPos() || active.empty()) {
			return true;
		} else {
			at(pos); // generate new items to visit
			return pos >= corners.size()
				|| (stopper && stopper->shouldStop(corners, pos + 1));
		}
	}

	/// Clone the Nexter instance for the specified collection.
	shared_ptr<Nexter> forColl(shared_ptr<Collection> coll) override {
		auto clone = std::make_shared<AboveMaxNexter>(coll);
		if (stopper) {
			clone->setStopper(stopper);
		}
		return clone;
	}

	/// Update the order to split between visited and unvisited objects
	void updateOrder(size_t pos, size_t choice) {
		if (pos >= corners.size()) {
			corners.push_back(choice);
		} else {
			corners.insert(corners.begin() + pos, choice);
		}
		coll->sort(choice);
		vector<size_t> newActive, newPending;
		newActive.reserve(active.size());
		newPending.reserve(pending.size());
		size_t maxCount = 0;
		for (auto& objlist : {active, pending}) {
			for (size_t obj : objlist) {
				size_t count = countNumBelow(coll, corners, obj);
				if (count > maxCount) {
					for (size_t obj2 : newActive) {
						newPending.push_back(obj2);
					}
					newActive.clear();
					newActive.push_back(obj);
					maxCount = count;
				} else if (count == maxCount) {
					newActive.push_back(obj);
				} else {
					newPending.push_back(obj);
				}
			}
		}

		// If maxCount == 1, then objects are not meaningfully above each other.
		// Note that all objects are above themselves.
		if (maxCount <= 1) {
			active.clear();
			pending.clear();
		} else {
			active.swap(newActive);
			pending.swap(newPending);
		}
	}

private:

	/// The collection we traverse over
	shared_ptr<Collection> coll;

	/// The traversal order so far
	vector<size_t> corners;

	/// Untraversed items with the max count
	vector<size_t> active;

	/// Untraversed items with less than the max count, but more than zero
	vector<size_t> pending;
};

/// Create a traversal which selects the point above the most points, with
/// respect to the convex hull of previously-visited points.
Traversal Traversal::AboveMax(shared_ptr<Collection> coll) {
	return Traversal(coll, std::make_shared<AboveMaxNexter>(coll));
}

/// Stops when a certain number of objects have been visited.
class MaxCountStopper : public Stopper {
public:

	/// Constructor.
	MaxCountStopper(size_t maxCount) : maxCount(maxCount) {}

	/// Decide whether to stop based on having visited the specified items.
	bool shouldStop(const vector<size_t>&, size_t num) override {
		return num > maxCount;
	}

private:

	/// The maximum number of visits to permit.
	const size_t maxCount;
};

/// Stop after some maximum number of items have been visited.
Traversal Traversal::stopAfter(size_t maxCount) {
	return addStopper(std::make_shared<MaxCountStopper>(maxCount));
}

/// Stops when the the next point is affinely dependent on the prior points.
/// That is, when it does not require some new dimension to be positioned
/// correctly.
class AffDepStopper : public Stopper {
public:

	/// Constructor.
	AffDepStopper(shared_ptr<Collection> coll) : coll(coll) {}

	/// Decide whether to stop based on having visited the specified items.
	bool shouldStop(const vector<size_t>& visited, size_t num) override {
		if (visited.size() <= 2) {
			return false;
		} else if (num > visited.size() + 1) {
			return true;
		}
		vector<size_t> corners;
		if (num > 1) {
			corners.resize(num - 1);
			std::copy_n(visited.begin(), num - 1, corners.begin());
		}
		return !isAffinelyIndependent(coll, visited[num-1], corners);
	}

private:

	/// Collection pointer.
	shared_ptr<Collection> coll;
};

/// Stop if the visited items are not affinely independent.
Traversal Traversal::stopWhenAffDep() {
	return addStopper(std::make_shared<AffDepStopper>(coll));
}

} // end namespace core
} // end namespace OGT_NAMESPACE
