/// @file  collection.cpp
/// @brief Definitions related to object collections.

#include <algorithm>
#include <array>
#include <map>
#include <ogt/core/collection.hpp>
#include <ogt/core/splitter.hpp>
#include <ogt/core/traversal.hpp>
#include <ogt/embed/cmp.hpp>
#include <ogt/util/random.hpp>
#include <queue>
#include <random>

using std::array;
using std::map;
using std::max;
using std::min;
using std::priority_queue;
using std::set;
using std::shared_ptr;
using std::vector;
using OGT_NAMESPACE::embed::CmpConstraint;
using OGT_NAMESPACE::util::random_choice;
using OGT_NAMESPACE::util::random_subset;

namespace OGT_NAMESPACE {
namespace core {

/// Implementation details of Collection not exposed in the header.
class CollectionImpl : public Collection {
public:

	/// Constructor.
	CollectionImpl(shared_ptr<Sorter> sorter, size_t nObjGlobal,
		vector<size_t> objs)
		: sorter(sorter)
		, nGlobal(nObjGlobal)
		, objs(objs)
		, orders()
		, ranks()
		, pairs()
	{
	}

	/// Construct a collection containing a subset of this one
	shared_ptr<Collection> subset(vector<size_t> items) const override {
		auto subcoll = std::make_shared<CollectionImpl>(sorter, nGlobal, items);
		subcoll->pairs = pairs;

		// Remember any stored order information
		if (!ranks.empty()) {
			set<size_t> sitems(items.begin(), items.end());
			for (const auto& pr : ranks) {
				if (sitems.find(pr.first) != sitems.end()) {
					subcoll->ranks[pr.first] = pr.second;
					subcoll->sort(pr.first, items);
					subcoll->orders[pr.first] = items;
				}
			}
		}

		return subcoll;
	}

	/// Get the number of objects in the global collection.
	size_t nObjGlobal() const override {
		return nGlobal;
	}

	/// Get the number of objects in the local collection.
	size_t nObjLocal() const override {
		return objs.size();
	}

	/// Get the ID of the nth object in the collection.
	size_t object(size_t n) const override {
		return objs[n];
	}

	/// Get the object IDs in the collection.
	const vector<size_t>& objects() const override {
		return objs;
	}

	/// Get a random object from the collection.
	size_t randObject() const override {
		return *random_choice(objs.begin(), objs.end());
	}

	/// Sort a pair of objects by distance to `a`.
	Cmp sortPair(size_t a, size_t b, size_t c) override {

		// Use basic properties
		if (b == c) return AB_EQ_AC;
		if (a == b) return AB_LT_AC;
		if (a == c) return AC_LT_AB;

		// Try to look up the object ranks
		const auto it = ranks.find(a);
		if (it != ranks.end()) {
			return (it->second[b] < it->second[c]) ? AB_LT_AC : AC_LT_AB;
		}

		// Consult the oracle, but try the pairs cache first
		const array<size_t,2> tails({min(b,c), max(b,c)});
		auto itp = pairs.find(a);
		if (itp == pairs.end()) {
			auto res = pairs.emplace(a, map<array<size_t,2>, CmpOutcome>());
			itp = res.first;
		} else {
			const auto itpp = itp->second.find(tails);
			if (itpp != itp->second.end()) {
				return itpp->second.outcome(b, c);
			}
		}
		Cmp result = sorter->sortPair(a, b, c);
		itp->second.emplace(tails, CmpOutcome(a, b, c, result));
		return result;
	}

	/// Sort the collection by distance from a head object, and store the
	/// results for future reference.
	void sort(size_t head) override {
		if (isSortedBy(head)) return;
		pairs.erase(head);
		vector<size_t> order(objs);
		sorter->sort(head, order);
		vector<size_t> rank(nGlobal, nGlobal);
		for (size_t pos = 0; pos < order.size(); pos++) {
			rank[order[pos]] = pos;
		}
		orders[head].swap(order);
		ranks[head].swap(rank);
	}

	/// Sort a vector of object indices in-place by distance to the head object.
	void sort(size_t head, std::vector<size_t>& objects) override {
		if (!isSortedBy(head)) {
			sort(head);
		}
		const auto it = ranks.find(head);
		if (it == ranks.end()) {
			throw UnsortedErr("Attempted to access ranks for unsorted head "
				+ std::to_string(head));
		}
		std::function<bool(size_t,size_t)> cmp =
			[&](size_t b, size_t c) -> bool {
				return it->second[b] < it->second[c];
			};
		std::sort(objects.begin(), objects.end(), cmp);
	}

	/// Sort the collection by distance to the head object, and place the
	/// corresponding comparisons into the provided vector.
	void sort(size_t head, vector<CmpConstraint>& cmps) override {
		if (!isSortedBy(head)) {
			sort(head);
		}
		const auto it = orders.find(head);
		for (size_t r1 = 1; r1 < it->second.size(); r1++) {
			const size_t c = it->second[r1];
			int r0 = r1 - 1;
			for (; r0 >= 1; r0--) {
				auto cmp = sorter->sortPair(head, it->second[r0], c);
				switch (cmp) {
				case AB_LT_AC:
					cmps.emplace_back(head, it->second[r0], c);
					// fall through

				case AC_NCMP:
					r0 = -1;
					// fall through

				default:
					break;
				}
			}
		}
	}

	/// Ask whether the collection has been sorted for a particular head yet.
	bool isSortedBy(size_t head) const override {
		return ranks.find(head) != ranks.end();
	}

	/// Get the rank of an object from a given head or throw UnsortedErr.
	size_t objRank(size_t head, size_t object) const override {
		const auto it = ranks.find(head);
		if (it == ranks.end()) {
			throw UnsortedErr("Attempted to access ranks for unsorted head "
				+ std::to_string(head));
		}
		return it->second[object];
	}

	/// Get the ranks of all objects from a given head or throw UnsortedErr.
	vector<size_t> objRanks(size_t head) const override {
		const auto it = ranks.find(head);
		if (it == ranks.end()) {
			throw UnsortedErr("Attempted to access ranks for unsorted head "
				+ std::to_string(head));
		}
		return it->second;
	}

	/// Get the object at a given rank from a given head or throw UnsortedErr.
	size_t objAt(size_t head, size_t rank) const override {
		const auto it = orders.find(head);
		if (it == orders.end()) {
			throw UnsortedErr("Attempted to access objects for unsorted head "
				+ std::to_string(head));
		}
		return it->second[rank];
	}

	/// Get the order of all objects from a given head or throw UnsortedErr.
	vector<size_t> objOrder(size_t head) const override {
		const auto it = orders.find(head);
		if (it == orders.end()) {
			throw UnsortedErr("Attempted to access order for unsorted head "
				+ std::to_string(head));
		}
		return it->second;
	}

	/// Get the objects with ranks `rank` and below.
	vector<size_t> objKnn(size_t head, size_t rank) override {
		const auto it = orders.find(head);
		if (it == orders.end()) {
			return sorter->objKnn(head, objs, rank);
		} else {
			auto first = it->second.begin() + 1;
			return vector<size_t>(first, first + rank);
		}
	}

	/// Get the object at maximum rank from a given head.
	size_t farthestFrom(size_t head) override {
		const auto it = orders.find(head);
		if (it != orders.end()) {
			return it->second.back();
		}
		size_t farthest = head;
		for (size_t obj : objs) {
			if (obj != head) {
				if (farthest == head) {
					farthest = obj;
				} else if (sortPair(head, farthest, obj) == AB_LT_AC) {
					farthest = obj;
				}
			}
		}
		return farthest;
	}

	/// Estimate the dimensionality of the set.
	size_t estimateDim(size_t maxObj) const override {
		auto sel = random_subset<size_t>(objs.begin(), objs.end(), maxObj);
		auto sub = subset(vector<size_t>(sel.begin(), sel.end()));
		size_t nIter = 0;
		auto order = sub->frftTraversal().stopWhenAffDep();
		for (auto it = order.begin(); it != order.end(); it++) {
			nIter++;
		}
		return nIter - 1;
	}

	/// A helper class for findEvenSubset()
	struct SubsetPt {
		size_t obj;
		size_t minSize;
		vector<array<size_t,2>> sizes;
		vector<size_t> ranks;

		/// Build a new instance and set its initial bucket sizes
		SubsetPt(const CollectionImpl* coll, const set<size_t>& refs,
			size_t obj)
			: obj(obj) {
			size_t maxRank = coll->nObjLocal() - 1;
			minSize = maxRank;
			for (size_t ref : refs) {
				size_t rank = coll->objRank(ref, obj);
				ranks.push_back(rank);
				sizes.push_back({rank - 1, maxRank - rank});
			}
		}

		/// Sorts in max min size order.
		bool operator<(const SubsetPt& other) const {
			return minSize > other.minSize;
		}

		/// Updates the min size based on new points added to the subset.
		void update(const map<size_t, set<size_t>>& refRanks) {
			size_t i = 0;
			for (const auto& pair : refRanks) {
				auto it = pair.second.lower_bound(ranks[i]);
				sizes[i][1] = *it - ranks[i];
				it--;
				sizes[i][0] = ranks[i] - *it;
				minSize = std::min(minSize, sizes[i][0]);
				minSize = std::min(minSize, sizes[i][1]);
				i++;
			}
		}
	};

	/// Find a subset of points which is evenly distributed through the
	/// collection.
	set<size_t> findEvenSubset(const set<size_t>& refs, size_t nPts) override {
		set<size_t> subset(refs);
		if (nObjLocal() <= nPts) {
			subset.insert(objs.begin(), objs.end());
			return subset;
		}

		// Prepare the initial list of ranks we're including for each ref
		map<size_t, set<size_t>> refRanks;
		for (size_t ref : refs) {
			auto& rs = refRanks[ref];
			for (size_t ref2 : refs) {
				rs.insert(objRank(ref, ref2));
			}
			rs.insert(nObjLocal());
		}

		// Prepare a priority queue for the points in the collection.
		priority_queue<SubsetPt> order;
		for (size_t obj : objects()) {
			if (subset.find(obj) == subset.end()) {
				SubsetPt pt(this, refs, obj);
				pt.update(refRanks);
				order.push(pt);
			}
		}

		// Add points from the queue until we have enough.
		while (subset.size() < nPts && !order.empty()) {
			SubsetPt choice = order.top();
			order.pop();
			choice.update(refRanks);
			while (!order.empty() && choice.minSize < order.top().minSize) {
				SubsetPt pt = order.top();
				order.pop();
				pt.update(refRanks);
				if (pt.minSize > choice.minSize) {
					order.push(choice);
					choice = pt;
				} else {
					order.push(pt);
				}
			}
			subset.insert(choice.obj);
			for (size_t ref : refs) {
				refRanks[ref].insert(objRank(ref, choice.obj));
			}
		}
		return subset;
	}

	/// Traverse in ascending ID order.
	Traversal ascIdTraversal() override {
		return Traversal::AscId(shared_from_this());
	}

	/// Traverse in random order.
	Traversal randomTraversal() override {
		return Traversal::Random(shared_from_this());
	}

	/// Traverse in FRFT order. The points traversed need to be sorted as we go,
	/// so this cannot be done with a const Collection.
	Traversal frftTraversal() override {
		return Traversal::FRFT(shared_from_this());
	}

	/// Traverse all points by distance to the head. Sorts by head.
	Traversal rankTraversal(size_t head) override {
		return Traversal::Rank(shared_from_this(), head);
	}

	/// Traverse the points in a lens by distance to pt1. Sorts by pt1 and pt2.
	Traversal lensTraversal(size_t pt1, size_t pt2) override {
		return Traversal::Lens(shared_from_this(), pt1, pt2);
	}

	/// Traverse the points in Above Max order.
	Traversal aboveMaxTraversal() override {
		return Traversal::AboveMax(shared_from_this());
	}

	/// Split into random subsets.
	/// @see Splitter::Random, RandomSplitter.
	vector<shared_ptr<Collection>> splitRandom(size_t nOverlap) override {
		return Splitter::Random(nOverlap)->split(shared_from_this());
	}

	/// Split into subsets of the kNN of visited objects.
	/// @see Splitter::Random, RandomSplitter.
	vector<shared_ptr<Collection>> splitTraversalKnn(Traversal order) override {
		return Splitter::Traversal(order)->split(shared_from_this());
	}

	/// Saves random true triples to the specified sink.
	void saveRandomTriples(std::ostream& out, size_t numTriples) override {
		if (objs.size() < 3) {
			return;
		}
		for (size_t i = 0; i < numTriples; i++) {
			size_t a = randObject();
			size_t b = randObject();
			while (b == a) {
				b = randObject();
			}
			size_t c = randObject();
			while (c == a || c == b) {
				c = randObject();
			}
			out << CmpOutcome(a, b, c, sortPair(a, b, c)) << std::endl;
		}
	}

	/// Returns the set of points within the ball centered on `center` and with
	/// radius `dist(center, radius)`.
	/// The parameter closed indicates whether the ball is open (excluding
	/// radius) or closed (including radius).
	/// If subset is not None, only points from subset which lie in the ball
	/// are returned.
	set<size_t> ball(size_t center, size_t radius, bool closed=true,
		const set<size_t>& subset=set<size_t>()) override {
        vector<size_t> result;
        vector<size_t>::iterator end;
        auto pred = [=](size_t x) {
        	return this->in_ball(x, center, radius, closed);
        };
        if (subset.empty()) {
        	result.resize(objs.size());
        	end = copy_if(objs.begin(), objs.end(), result.begin(), pred);
        } else {
        	result.resize(subset.size());
        	end = copy_if(subset.begin(), subset.end(), result.begin(), pred);
        }
        return set<size_t>(result.begin(), end);
	}

	/// Determines whether point(s) `x` lie within the ball centered on `center`
	/// and with radius `dist(center, radius)`.
	bool in_ball(size_t x, size_t center, size_t radius, bool closed=true)
	override {
		auto result = sortPair(center, x, radius);
		if (result != AB_LT_AC && closed) {
			return sortPair(center, radius, x) != AB_LT_AC;
		}
		return result == AB_LT_AC;
	}

	/// Returns the number of points in lens(p, q).
	size_t lens_size(size_t p, size_t q, bool closed=true) override {
		size_t count = 0;
		for (size_t x : objs) {
			if (in_lens(x, p, q, closed)) {
				count++;
			}
		}
		return count;
	}

	/// Return a set containing the points x for which edge pq is the longest in
	/// triangle xpq.
	set<size_t> lens(size_t p, size_t q, bool closed=true, 
		const set<size_t>& subset=set<size_t>()) override {
		set<size_t> result;
		if (subset.empty()) {
			for (size_t x : objs) {
				if (in_lens(x, p, q, closed)) {
					result.insert(x);
				}
			}
		} else {
			for (size_t x : subset) {
				if (in_lens(x, p, q, closed)) {
					result.insert(x);
				}
			}
		}
		return result;
	}

	/// Asks whether x is in lens(p, q).
	bool in_lens(size_t x, size_t p, size_t q, bool closed=true) override {
		return in_ball(x, p, q, closed) && in_ball(x, q, p, closed);
	}

private:

	/// The sorter which defines order on the collection.
	shared_ptr<Sorter> sorter;

	/// The number of objects in the global collection.
	const size_t nGlobal;

	/// The local object collection.
	const vector<size_t> objs;

	/// The collection order when sorted by distance from various heads.
	/// orders[head][rank] gives the entry from objs at the specified rank.
	map<size_t, vector<size_t>> orders;

	/// Object ranks when sorted by distance from various heads.
	/// ranks[head][obj] gives the rank of the object.
	map<size_t, vector<size_t>> ranks;

	/// Sorted pairs stored for heads which are not yet fully sorted.
	map<size_t, map<array<size_t,2>, CmpOutcome>> pairs;
};

// Construct a collection with object IDs 0 to nObj-1.
shared_ptr<Collection> Collection::Create(shared_ptr<Sorter> sorter,
	size_t nObj) {
	vector<size_t> obj(nObj);
	std::iota(obj.begin(), obj.end(), 0);
	return std::make_shared<CollectionImpl>(sorter, nObj, obj);
}

/// Shorthand for creating a collection with an OracleSorter.
shared_ptr<Collection> Collection::Create(shared_ptr<Oracle> oracle,
	size_t nObj) {
	return Collection::Create(OracleSorter::Create(oracle), nObj);
}

// Construct a collection with a subset of IDs from some global collection.
shared_ptr<Collection> Collection::Create(shared_ptr<Sorter> sorter,
	size_t nObj, vector<size_t> obj) {
	std::sort(obj.begin(), obj.end());
	return std::make_shared<CollectionImpl>(sorter, nObj, obj);
}

/// Construct a collection with a subset of IDs from some global collection.
shared_ptr<Collection> Collection::Create(shared_ptr<Oracle> oracle,
	size_t nObj, vector<size_t> obj) {
	std::sort(obj.begin(), obj.end());
	return Collection::Create(OracleSorter::Create(oracle), nObj, obj);
}

} // end namespace core
} // end namespace OGT_NAMESPACE
