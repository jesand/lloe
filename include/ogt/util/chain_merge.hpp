/// @file  chain_merge.hpp
/// @brief Declarations related to the Chain Merge algorithm.

#pragma once
#ifndef OGT_UTIL_CHAIN_MERGE_HPP
#define OGT_UTIL_CHAIN_MERGE_HPP

#include <ogt/config.hpp>
#include <ogt/core/oracle.hpp>
#include <string>

using OGT_NAMESPACE::core::Cmp;
using OGT_NAMESPACE::core::Oracle;

namespace OGT_NAMESPACE {
namespace util {

/// An exception thrown when impossible sort operations are requested.
struct SortErr : public std::runtime_error {
	SortErr(std::string msg) : std::runtime_error(msg) {}
};

/// A comparison function used by ChainMerge.
typedef std::function<Cmp(size_t,size_t)> CMOracle;

/// Create a CMOracle from an Oracle, fixing the specified head
std::shared_ptr<CMOracle> createCMOracleWithHead(std::shared_ptr<Oracle> oracle,
	size_t head);

/// An efficient container for sorting partially ordered data and performing
/// constant-time lookups once sorted.
///
/// Citations for base sort and peeling algorithms:
/// [1] C. Daskalakis, R. M. Karp, E. Mossel, S. Riesenfeld, and E. Verbin,
///     "Sorting and Selection in Posets," arXiv.org, vol. cs.DS. 10-Jul-2007.
struct ChainMerge {

	/// Create a new instance
	static std::shared_ptr<ChainMerge> Create();

	/// Virtual destructor
	virtual ~ChainMerge() = default;

	/// Remove all objects.
	virtual void clear() = 0;

	/// Sort objects in the range from 'begin' to just before 'end.'
	virtual void sort(size_t* begin, size_t* end,
		std::shared_ptr<CMOracle> oracle) = 0;

	/// Look up the ordinal relationship between two previously-sorted objects.
	virtual Cmp compare(size_t b, size_t c) = 0;

	/// Reduce the number of chains to a minimum, and report their number.
	virtual size_t compact() = 0;

	/// Add the specified objects to the list, if they are new.
	/// Returns the number of objects added.
	virtual size_t addObjects(size_t* begin, size_t* end) = 0;

	/// Add a known comparison to the list
	virtual void addCmp(size_t b, size_t c, Cmp order) = 0;
};

} // end namespace util
} // end namespace OGT_NAMESPACE
#endif /* OGT_UTIL_CHAIN_MERGE_HPP */
