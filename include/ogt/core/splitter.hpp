/// @file  splitter.hpp
/// @brief Declarations related to subdividing object collections.

#pragma once
#ifndef OGT_CORE_SPLITTER_HPP
#define OGT_CORE_SPLITTER_HPP

#include <ogt/config.hpp>
#include <memory>
#include <vector>

namespace OGT_NAMESPACE {
namespace core {

struct Collection; // Forward declaration
class Traversal; // Forward declaration

/// Divides a collection into overlapping subsets. This is done because it is
/// sometimes helpful to work on smaller subproblems.
struct Splitter {

	/// Virtual destructor
	virtual ~Splitter() = default;

	/// Split the collection.
	virtual std::vector<std::shared_ptr<Collection>> split(
		std::shared_ptr<Collection> coll) = 0;

	/// Create a random splitter. This chooses O(sqrt(n)) subsets of
	/// O(sqrt(n)) objects by selecting at random, where n is the global
	/// collection size.
	/// The set is first partitioned, and then nOverlap members from previous
	/// subsets are randomly chosen to add to later subsets.
	/// @see RandomSplitter
	static std::shared_ptr<Splitter> Random(size_t nOverlap);

	/// Create a traversal-based splitter. This chooses O(sqrt(n)) subsets of
	/// O(sqrt(n)) objects,, where n is the global collection size, by
	/// traversing in the specified order and choosing the nearest sqrt(n)
	/// objects to each visited object.
	/// @see TraversalSplitter
	static std::shared_ptr<Splitter> Traversal(Traversal order);
};

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_SPLITTER_HPP */
