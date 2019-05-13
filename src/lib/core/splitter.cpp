/// @file  splitter.cpp
/// @brief Definitions related to subdividing object collections.

#include <ogt/core/splitter.hpp>
#include <ogt/core/collection.hpp>
#include <ogt/core/traversal.hpp>
#include <ogt/util/random.hpp>
#include <algorithm>
#include <cmath>

using OGT_NAMESPACE::util::globalRand;
using OGT_NAMESPACE::util::random_subset;
using std::ceil;
using std::make_shared;
using std::set;
using std::shared_ptr;
using std::sqrt;
using std::vector;

namespace OGT_NAMESPACE {
namespace core {

/// A Collection Splitter which selects subsets at random.
/// @see Splitter::Random().
class RandomSplitter : public Splitter {
public:

	/// Constructor.
	RandomSplitter(size_t nOverlap) : nOverlap(nOverlap) {}

	/// Split the collection.
	vector<shared_ptr<Collection>> split(shared_ptr<Collection> coll) override {

		// Permute the objects at random
		auto objects = coll->objects();
		auto& rng = globalRand();
		std::shuffle(objects.begin(), objects.end(), rng);

		// Choose the subsets, with random overlap
		size_t nObj = objects.size();
		size_t nSub = ceil(sqrt(nObj));
		vector<shared_ptr<Collection>> subsets;
		for (size_t start = 0; start < nObj; start += nSub) {
			auto itStart = objects.begin() + start;
			vector<size_t> subset(itStart, min(itStart + nSub, objects.end()));
			if (nOverlap > 0) {
				auto overlap = random_subset<size_t>(objects.begin(), itStart,
					nOverlap);
				for (size_t pt : overlap) {
					subset.push_back(pt);
				}
			}
			subsets.emplace_back(coll->subset(subset));
		}
		return subsets;
	}

private:

	/// The number of overlapping points to include in later subsets.
	const size_t nOverlap;
};

/// Create a random splitter.
shared_ptr<Splitter> Splitter::Random(size_t nOverlap) {
	return make_shared<RandomSplitter>(nOverlap);
}

/// A Collection Splitter which selects subsets of objects near objects visisted
/// in some traversal order.
/// @see Splitter::Traversal.
class TraversalSplitter : public Splitter {
public:

	/// Constructor.
	TraversalSplitter(class Traversal order) : order(order) {}

	/// Split the collection.
	vector<shared_ptr<Collection>> split(shared_ptr<Collection> coll) override {
		size_t nObj = coll->nObjLocal();
		size_t nSub = ceil(sqrt(nObj));
		const auto& objs = coll->objects();
		set<size_t> left(objs.begin(), objs.end());
		vector<shared_ptr<Collection>> subsets;
		auto co = order.forColl(coll);
		for (auto it = co.begin(); it != co.end(); it++) {
			auto knn = coll->objKnn(*it, nSub);
			knn.push_back(*it);
			for (auto neighbor : knn) {
				it = co.visit(it, neighbor);
				left.erase(neighbor);
			}
			subsets.emplace_back(coll->subset(knn));
			if (left.empty()) {
				break;
			}
		}
		return subsets;
	}

private:

	/// The traversal order.
	class Traversal order;
};

/// Create a traversal-based splitter.
shared_ptr<Splitter> Splitter::Traversal(class Traversal order) {
	return make_shared<TraversalSplitter>(order);
}

} // end namespace core
} // end namespace OGT_NAMESPACE
