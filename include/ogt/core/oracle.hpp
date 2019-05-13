/// @file  oracle.hpp
/// @brief Declarations related to comparison oracles.

#pragma once
#ifndef OGT_CORE_ORACLE_HPP
#define OGT_CORE_ORACLE_HPP

#include <ogt/config.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

namespace OGT_NAMESPACE {
namespace core {

/// The possible outcomes of a triple comparison.
enum Cmp {

	// dist(a,b) < dist(a,c)
	AB_LT_AC,

	// dist(a,c) < dist(a,b)
	AC_LT_AB,

	// dist(a,b) = dist(a,c)
	AB_EQ_AC,

	// dist(a,b) cannot be measured, but dist(a,c) can
	AB_NCMP,

	// dist(a,c) cannot be measured, but dist(a,b) can
	AC_NCMP,

	// comparison result cannot be determined for this triple
	AB_AC_NCMP
};

/// Convert a comparison outcome to a string representation
std::string to_string(Cmp cmp);

/// Convert a comparison outcome from a string representation or throw
/// std::invalid_argument.
Cmp stocmp(std::string str);

/// A comparison outcome.
struct CmpOutcome {

	/// Default constructor
	CmpOutcome() {}

	/// Constructor
	CmpOutcome(size_t a, size_t b, size_t c, Cmp cmp)
		: a(a), b(b), c(c), cmp(cmp) {}

	/// Equality test
	bool operator==(const CmpOutcome& copy) const {
		return a == copy.a && b == copy.b && c == copy.c && cmp == copy.cmp;
	}

	/// Give the outcome with the specified order for b and c
	Cmp outcome(size_t b, size_t c) {
		if (b == c) {
			return AB_EQ_AC;
		} else if (b == this->b && c == this->c) {
			return cmp;
		} else if (c == this->b && b == this->c) {
			switch(cmp) {
			case AB_LT_AC:
				return AC_LT_AB;
			case AB_NCMP:
				return AC_NCMP;
			case AC_LT_AB:
				return AB_LT_AC;
			case AC_NCMP:
				return AB_NCMP;
			case AB_EQ_AC:
			case AB_AC_NCMP:
				return cmp;
			}
		}
		return AB_AC_NCMP; // answer not inferable from this comparison
	}

	size_t a, b, c;
	Cmp cmp;
};

/// Stream a comparison to output
std::ostream& operator<<(std::ostream& os, const CmpOutcome& cmp);

/// Stream a comparison from input
std::istream& operator>>(std::istream& is, CmpOutcome& cmp);

/// A comparison oracle. Inputs a triple (a,b,c) and returns a comparison of
/// dist(a, b) and dist(a, c).
typedef std::function<Cmp(size_t,size_t,size_t)> Oracle;

/// Randomly flip a certain fraction of triples.
std::shared_ptr<Oracle> createUniformNoisyOracle(std::shared_ptr<Oracle> oracle,
	double probFlipped);

/// Create an oracle which invokes the specified function whenever it is called.
/// This is useful for logging or storing the raw comparisons.
template<typename F>
std::shared_ptr<Oracle> createLoggingOracle(std::shared_ptr<Oracle> oracle,
	F& logger) {
	return std::make_shared<Oracle>([=](auto a, auto b, auto c) -> Cmp {
		Cmp result = (*oracle)(a, b, c);
		logger(CmpOutcome(a, b, c, result));
		return result;
	});
}

/// Create an oracle which logs all comparisons to an ostream and counts the
/// number of comparisons streamed.
std::shared_ptr<Oracle> createStreamingOracleWithCounter(
	std::shared_ptr<Oracle> oracle, std::ostream& os,
	std::shared_ptr<size_t> counter);

/// Create an oracle which logs all comparisons to an ostream.
std::shared_ptr<Oracle> createStreamingOracle(std::shared_ptr<Oracle> oracle,
	std::ostream& os);

/// Get an oracle which relies on known pairwise distances.
/// Distances are considered equal if they differ by no more than eps.
/// The parameter dist must be callable and have type (size_t, size_t) -> number,
/// where number is any type which can be cast to double. It gives
/// the distance between two arbitrary points in the collection, or 0 for unknown
/// distances. Its two parameters will never have the same value.
template<typename distfn>
std::shared_ptr<Oracle> createDistOracle(const distfn& dist, double eps=0) {
	return std::make_shared<Oracle>([&dist,eps] (auto a, auto b, auto c) -> Cmp {
		if (b == c) {
			return AB_EQ_AC;
		} else if (a == b) {
			return AB_LT_AC;
		} else if (a == c) {
			return AC_LT_AB;
		}
		double ab = static_cast<double>(dist(a,b));
		double ac = static_cast<double>(dist(a,c));
		if (ab == 0 || !std::isfinite(ab)) {
			if (ac == 0 || !std::isfinite(ac)) {
				return AB_AC_NCMP;
			}
			return AB_NCMP;
		} else if (ac == 0 || !std::isfinite(ac)) {
			return AC_NCMP;
		} else if (ab + eps < ac) {
			return AB_LT_AC;
		} else if (ac + eps < ab) {
			return AC_LT_AB;
		} else {
			return AB_EQ_AC;
		}
	});
}

/// Get an oracle which relies on the specified distance and position functions.
template<typename posfn, typename distfn>
std::shared_ptr<Oracle> createPosOracle(posfn pos, distfn dist, double eps=0) {
	return std::make_shared<Oracle>([=] (size_t a, size_t b, size_t c) -> Cmp {
		if (b == c) {
			return AB_EQ_AC;
		} else if (a == b) {
			return AB_LT_AC;
		} else if (a == c) {
			return AC_LT_AB;
		}
		auto xa = pos(a);
		auto xb = pos(b);
		auto xc = pos(c);
		double ab = static_cast<double>(dist(xa,xb));
		double ac = static_cast<double>(dist(xa,xc));
		if (ab == 0) {
			if (ac == 0) {
				return AB_AC_NCMP;
			}
			return AB_NCMP;
		} else if (ac == 0) {
			return AC_NCMP;
		} else if (ab + eps < ac) {
			return AB_LT_AC;
		} else if (ac + eps < ab) {
			return AC_LT_AB;
		} else {
			return AB_EQ_AC;
		}
	});
}

/// Available distance metrics
enum DistFn {
	EUCLIDEAN_DIST,
	COSINE_DIST,
	JACCARD_DIST
};

/// Convert a comparison outcome to a string representation
std::string to_string(DistFn distFn);

/// Get a DistFn for a string or throw std::invalid_argument.
DistFn stodistfn(std::string str);

/// Get an oracle which uses Euclidean distance between rows of the
/// specified position matrix.
template<typename T>
std::shared_ptr<Oracle> createEuclideanPosOracle(T& pos, double eps=0) {
	return createPosOracle(
		[&pos](auto idx) { return pos.row(idx); },
		[](auto a, auto b) { return (a - b).norm(); },
		eps);
}

/// Get an oracle which uses Cosine distance between rows of the
/// specified position matrix.
template<typename T>
std::shared_ptr<Oracle> createCosinePosOracle(T& pos, double eps=0) {
	return createPosOracle(
		[&pos](auto idx) { return pos.row(idx); },
		[](auto a, auto b) {
			return 1.0 - (a.cwiseProduct(b).sum() / (a.norm() * b.norm()));
		},
		eps);
}

/// Get an oracle which uses Jaccard distance between rows of the
/// specified position matrix.
template<typename M, typename V, typename IT>
std::shared_ptr<Oracle> createJaccardPosOracle(M& pos, double eps=0) {
	return createPosOracle(
		[&pos](auto idx) -> V { return pos.row(idx); },
		[](V a, V b) {
			IT ita(a), itb(b);
			double inBoth = 0, total = 0;
			while (ita && itb) {
				total++;
				if (ita.index() < itb.index()) {
					++ita;
				} else if (ita.index() > itb.index()) {
					++itb;
				} else {
					inBoth++;
					++ita;
					++itb;
				}
			}
			for (; ita; ++ita) {
				total++;
			}
			for (; itb; ++itb) {
				total++;
			}

			return total ? (total - inBoth) / total : INFINITY;
		},
		eps);
}

/// Create an oracle which responds with cached comparisons sourced from a
/// stream.
/// @see createStreamingOracle, IstreamOracle
std::shared_ptr<Oracle> createIstreamOracle(std::istream& is);

/// Create an oracle which responds with cached comparisons sourced from a
/// stream. The comparisons must be presented in rank order, for fully-sorted
/// lists.
/// @see createStreamingOracle, IstreamOracle
std::shared_ptr<Oracle> createIstreamRankOracle(std::istream& is);

/// An object capable of sorting a subset of objects by increasing distance
/// (or, equivalently, decreasing similarity) to a "head" object.
/// This is generalized to support arbitrary implementations, e.g. to handle
/// noisy comparisons or to sort via a crowdsourcing operation.
struct Sorter {

	/// Sort a pair of objects by distance to `a`.
	/// This may be an invocation of an oracle, or may be some more complicated
	/// inference method.
	virtual Cmp sortPair(size_t a, size_t b, size_t c) = 0;

	/// Sort the objects in-place by increasing distance to the head object.
	virtual void sort(size_t head, std::vector<size_t>& objects) = 0;

	/// Get the objects with ranks `rank` and below.
	virtual std::vector<size_t> objKnn(size_t head,
		std::vector<size_t> objects, size_t rank) = 0;

	/// Virtual destructor.
	virtual ~Sorter() = default;
};

/// A Sorter which uses an Oracle to compare objects.
struct OracleSorter : public Sorter {

	/// Build an oracle-based sorter.
	static std::shared_ptr<OracleSorter> Create(std::shared_ptr<Oracle> oracle);

	/// Virtual destructor.
	virtual ~OracleSorter() = default;
};

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_ORACLE_HPP */
