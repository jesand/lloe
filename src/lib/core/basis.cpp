/// @file  basis.cpp
/// @brief Definitions related to orthogonal bases for collections.

#include <ogt/core/basis.hpp>
#include <ogt/core/collection.hpp>
#include <ogt/core/core.hpp>
#include <ogt/core/traversal.hpp>
#include <sstream>

using std::shared_ptr;
using std::weak_ptr;
using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace OGT_NAMESPACE {
namespace core {

/// A concrete implementation of a Basis
class BasisImpl : public Basis {
public:

	/// Create an empty basis
	BasisImpl(weak_ptr<Collection> coll)
		: collptr(coll)
	{
	}

	/// Get the number of axes in the basis.
	size_t numAxes() const override {
		return axes.size();
	}

	/// Get the number of points along a specified axis.
	size_t axisLength(size_t axis) const override {
		assert(axis < axes.size());
		return axes[axis].size();
	}

	/// Get the points along a specified axis
	const vector<size_t>& axis(size_t axis) const override {
		assert(axis < axes.size());
		return axes[axis];
	}

	/// Read a basis from an istream
	void read(std::istream& is) {
		axes.clear();
		std::string line;
		while (std::getline(is, line)) {
			std::stringstream ss(line);
			vector<size_t> axis;
			size_t pt;
			char comma;
			while (ss >> pt) {
				axis.push_back(pt);
				ss >> comma;
			}
			axes.push_back(axis);
		}
	}

	/// Generate a new basis for the collection.
	void generate() {
		auto coll = collptr.lock();
		if (!coll) {
			throw std::runtime_error(
				"Called BasisImpl::generate() with a null Collection pointer");
		}
		axes.clear();
		vector<size_t> corners;
		auto order = coll->aboveMaxTraversal().stopWhenAffDep();
		for (auto iter = order.begin(); iter != order.end(); iter++) {
			size_t pt1 = *iter;
			coll->sort(pt1);

			// Choose the farthest point which pt1 is above as pt2
			size_t pt2 = pt1;
			size_t maxRank = 0;
			if (corners.empty()) {
				pt2 = coll->farthestFrom(pt1);
			} else {
				for (auto pt : lens(coll, corners, pt1)) {
					size_t rank = coll->objRank(pt1, pt);
					if (rank > maxRank) {
						maxRank = rank;
						pt2 = pt;
					}
				}
			}
			if (pt1 == pt2) {
				throw std::runtime_error(
					"Got a point in an aboveMaxTraversal() that was "
					"not above anything");
			}
			iter = order.visit(iter, pt2);
			coll->sort(pt2);

			auto axis = nearLine(coll, pt1, pt2);
			axes.push_back(axis);
			corners.push_back(pt1);
			corners.push_back(pt2);
		}
	}

	/// Get an object's position within the basis.
	VectorXd posFromLensMedianIndex(size_t object) const override {
		auto coll = collptr.lock();
		if (!coll) {
			throw std::runtime_error(
				"Called BasisImpl::posFromLensMedianIndex() with a null "
				"Collection pointer");
		}
		VectorXd pos(axes.size());
		for (size_t ai = 0; ai < axes.size(); ai++) {
			const auto& axis = axes[ai];
			const size_t pt1 = axis.front(), pt2 = axis.back();
			size_t fromIdx = axis.size(), toIdx = 0;
			const size_t r1 = coll->objRank(pt1, object);
			const size_t r2 = coll->objRank(pt2, object);
			for (size_t idx = 0; idx < axis.size(); idx++) {
				if (coll->objRank(pt1, axis[idx]) <= r1
					&& coll->objRank(pt2, axis[idx]) <= r2) {
					if (idx < fromIdx) {
						fromIdx = idx;
					}
					if (idx > toIdx) {
						toIdx = idx;
					}
				}
			}
			pos(ai) = (fromIdx + toIdx) / 2.0;
		}
		return pos;
	}

	/// Embed the collection using a basis.
	MatrixXd embedFromLensMedianIndex() override {
		auto coll = collptr.lock();
		if (!coll) {
			throw std::runtime_error(
				"Called BasisImpl::embedFromLensMedianIndex() with a null "
				"Collection pointer");
		}

		// Sort by axis endpoints, if necessary
		for (const auto& axis : axes) {
			coll->sort(axis.front());
			coll->sort(axis.back());
		}

		// Embed the collection
		MatrixXd X(coll->nObjGlobal(), axes.size());
		for (size_t object : coll->objects()) {
			X.row(object) = posFromLensMedianIndex(object);
		}
		return X;
	}

private:

	/// The Collection to which this basis applies.
	const weak_ptr<Collection> collptr;

	/// The basis axes.
	vector<vector<size_t>> axes;
};

/// Read a basis from an istream.
shared_ptr<Basis> Basis::Read(shared_ptr<Collection> coll, std::istream& is) {
	auto basis = std::make_shared<BasisImpl>(coll);
	basis->read(is);
	return basis;
}

/// Generate a basis for a collection, using comparisons as needed.
shared_ptr<Basis> Basis::Create(shared_ptr<Collection> coll) {
	auto basis = std::make_shared<BasisImpl>(coll);
	basis->generate();
	return basis;
}

/// Stream a string representation of a basis to an ostream
std::ostream& operator<<(std::ostream& os, const Basis& basis) {
	const size_t nAxes = basis.numAxes();
	for (size_t axis = 0; axis < nAxes; axis++) {
		const auto& points = basis.axis(axis);
		os << points[0];
		for (size_t i = 1; i < points.size(); i++) {
			os << "," << points[i];
		}
		os << std::endl;
	}
	return os;
}

} // end namespace core
} // end namespace OGT_NAMESPACE
