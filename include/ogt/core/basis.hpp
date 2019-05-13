/// @file  basis.hpp
/// @brief Declarations related to near-orthogonal bases of collections

#pragma once
#ifndef OGT_CORE_BASIS_HPP
#define OGT_CORE_BASIS_HPP

#include <ogt/config.hpp>
#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <vector>

namespace OGT_NAMESPACE {
namespace core {

struct Collection; // Forward declaration

/// A basis, consisting of a set of nearly-orthogonal axes.
/// Each axis is a set of points near the line connecting two endpoints.
struct Basis {
	virtual ~Basis() = default;

	/// Read a basis for a given Collection from an istream.
	static std::shared_ptr<Basis> Read(std::shared_ptr<Collection> coll,
		std::istream& is);

	/// Generate a basis for a collection, using comparisons as needed.
	static std::shared_ptr<Basis> Create(std::shared_ptr<Collection> coll);

	/// Get the number of axes in the basis.
	virtual size_t numAxes() const = 0;

	/// Get the number of points along a specified axis.
	virtual size_t axisLength(size_t axis) const = 0;

	/// Get the points along a specified axis
	virtual const std::vector<size_t>& axis(size_t axis) const = 0;

	/// Get an object's position within the basis.
	/// This method selects the index of the median axis point within the lens
	/// beneath the object as its coordinate on an axis.
	virtual Eigen::VectorXd posFromLensMedianIndex(size_t object) const = 0;

	/// Embed the collection using a basis.
	virtual Eigen::MatrixXd embedFromLensMedianIndex() = 0;
};

/// Stream a string representation of a basis to an ostream
std::ostream& operator<<(std::ostream& os, const Basis& basis);

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_BASIS_HPP */
