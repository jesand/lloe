/// @file  oracle.cpp
/// @brief Definitions related to comparison oracles.

#include <ogt/core/oracle.hpp>
#include <ogt/util/chain_merge.hpp>
#include <ogt/util/random.hpp>
#include <ogt/io/io.hpp>
#include <algorithm>
#include <map>
#include <timsort.hpp>

using OGT_NAMESPACE::io::IoErr;
using OGT_NAMESPACE::util::ChainMerge;
using OGT_NAMESPACE::util::random_choice;
using OGT_NAMESPACE::util::random_coin_flip;
using std::map;
using std::max;
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;

namespace OGT_NAMESPACE {
namespace core {

/// Convert a comparison outcome to a string representation.
string to_string(Cmp cmp) {
	switch (cmp) {
		case AB_LT_AC:
			return "<";
		case AC_LT_AB:
			return ">";
		case AB_EQ_AC:
			return "=";
		case AB_NCMP:
			return "nab";
		case AC_NCMP:
			return "nac";
		case AB_AC_NCMP:
			return "nabc";
	}
	return "<invalid>";
}

/// Convert a comparison outcome from a string representation or throw
/// std::invalid_argument.
Cmp stocmp(string str) {
	if (str == "<") {
		return AB_LT_AC;
	} else if (str == ">") {
		return AC_LT_AB;
	} else if (str == "=") {
		return AB_EQ_AC;
	} else if (str == "nab") {
		return AB_NCMP;
	} else if (str == "nac") {
		return AC_NCMP;
	} else if (str == "nabc") {
		return AB_AC_NCMP;
	}
	throw std::invalid_argument("No Cmp value has string \"" + str + "\"");
}

/// Stream a comparison to output
std::ostream& operator<<(std::ostream& os, const CmpOutcome& cmp) {
	os << cmp.a << "," << cmp.b << "," << cmp.c << "," << to_string(cmp.cmp);
	return os;
}

/// Stream a comparison from input
std::istream& operator>>(std::istream& is, CmpOutcome& cmp) {
	string str;
	char comma; // ignored; may not be a comma
	is >> cmp.a >> comma >> cmp.b >> comma >> cmp.c >> comma >> str;
	if (is) {
		try {
			cmp.cmp = stocmp(str);			
		} catch(const std::invalid_argument&) {
			if (static_cast<size_t>(atoi(str.c_str())) == cmp.a) {
				cmp.cmp = AB_LT_AC;
			} else {
				throw;
			}
		}
	}
	return is;
}

/// Randomly flip a certain fraction of triples.
shared_ptr<Oracle> createUniformNoisyOracle(shared_ptr<Oracle> oracle,
	double probFlipped) {
	return std::make_shared<Oracle>([=](auto a, auto b, auto c) -> Cmp {
		Cmp result = (*oracle)(a, b, c);
		switch (result) {
			case AB_LT_AC:
				return random_coin_flip(probFlipped) ? AC_LT_AB : result;
			case AC_LT_AB:
				return random_coin_flip(probFlipped) ? AB_LT_AC : result;
			default:
				return result;
		}
	});
}

/// Create an oracle which logs all comparisons to an ostream.
shared_ptr<Oracle> createStreamingOracle(shared_ptr<Oracle> oracle,
	std::ostream& os) {
	shared_ptr<size_t> counter;
	return createStreamingOracleWithCounter(oracle, os, counter);
}

/// Create an oracle which logs all comparisons to an ostream and counts the
/// number of comparisons streamed.
shared_ptr<Oracle> createStreamingOracleWithCounter(shared_ptr<Oracle> oracle,
	std::ostream& os, shared_ptr<size_t> counter)
 {
	auto streamer = [&os, counter](CmpOutcome cmp) {
		if (counter) {
			*counter += 1;
		}
		os << cmp << std::endl;
	};
	return createLoggingOracle(oracle, streamer);
}


/// Private implementation of OracleSorter so changing implementation details
/// don't force a large recompile.
class OracleSorterImpl : public OracleSorter {
public:

	/// Create an instance.
	OracleSorterImpl(shared_ptr<Oracle> oracle)
		: oracle(oracle)
	{
	}

	// Sort a pair of objects by distance to `a`.
	inline Cmp sortPair(size_t a, size_t b, size_t c) override {
		if (b == c) return AB_EQ_AC;
		if (a == b) return AB_LT_AC;
		if (a == c) return AC_LT_AB;
		return (*oracle)(a, b, c);
	}

	struct LTHead {
		const size_t head;
		shared_ptr<Oracle> oracle;
		LTHead(size_t head, shared_ptr<Oracle> oracle) : head(head), oracle(oracle) {}
		bool operator()(size_t b, size_t c) const {
			if (b == c) return false;
			if (head == b) return true;
			if (head == c) return false;
			return (*oracle)(head, b, c) == AB_LT_AC;
		}
	};

	/// Sort the objects in-place by increasing distance to the head object.
	void sort(size_t head, vector<size_t>& objects) override {
		LTHead cmp(head, oracle);
		gfx::timsort(objects.begin(), objects.end(), cmp);
	}

	/// Get the objects with ranks `rank` and below.
	vector<size_t> objKnn(size_t head, vector<size_t> objects, size_t rank)
		override {
		auto it = std::find(objects.begin(), objects.end(), head);
		if (it != objects.end()) {
			objects.erase(it);
		}
		return objKnnHelper(head, objects.begin(), objects.end(), rank);
	}

	/// A helper for objKnn().
	vector<size_t> objKnnHelper(size_t head, vector<size_t>::iterator begin,
		vector<size_t>::iterator end, size_t rank) {
		if (static_cast<size_t>(end - begin) <= rank) {
			return vector<size_t>(begin, end);
		}
		size_t pivot = *random_choice(begin, end);
		std::function<bool(size_t)> cmp =
			[=](size_t obj) -> bool {
				return sortPair(head, obj, pivot) == AB_LT_AC;
			};
		auto it = std::partition(begin, end, cmp);
		size_t itRank = it - begin; // num els in first partition half
		if (itRank < rank) {
			vector<size_t> res(begin, it);
			res.reserve(rank);
			for (auto obj : objKnnHelper(head, it, end, rank - itRank)) {
				res.push_back(obj);
			}
			return res;
		} else if (itRank > rank) {
			return objKnnHelper(head, begin, it, rank);
		} else {
			return vector<size_t>(begin, it);
		}
	}

private:

	/// The oracle used for sorting.
	shared_ptr<Oracle> oracle;
};

/// Build an oracle-based sorter.
std::shared_ptr<OracleSorter> OracleSorter::Create(
	std::shared_ptr<Oracle> oracle) {
	return std::make_shared<OracleSorterImpl>(oracle);
}

} // end namespace core
} // end namespace OGT_NAMESPACE
