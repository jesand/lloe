/// @file  dists.hpp
/// @brief Generating distributions for synthetic datasets.

#pragma once
#ifndef OGT_UTIL_DISTS_HPP
#define OGT_UTIL_DISTS_HPP

#include <ogt/config.hpp>
#include <array>
#include <Eigen/Dense>
#include <json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace OGT_NAMESPACE {
namespace util {

/// The type thrown when invalid methods are reqested.
struct InvalidPointGeneratorErr : public std::runtime_error {

	/// Build an InvalidPointGeneratorErr with the specified message.
	InvalidPointGeneratorErr(std::string message)
		: std::runtime_error("ogt::InvalidPointGeneratorErr: " + message) {
	}
};

/// An abstract generator of datasets.
struct PointGenerator {
	virtual ~PointGenerator() = default;

	/// Create a generator using the specified method, specified in JSON. Throws
	/// InvalidPointGeneratorErr if method is invalid.
	static std::shared_ptr<PointGenerator> Create(nlohmann::json method);

	/// Generate a batch of points.
	virtual Eigen::MatrixXd generate(size_t nPoints) = 0;
};

/// Generate points uniformly from a box.
struct UnifBoxDist : PointGenerator {
	virtual ~UnifBoxDist() = default;

	/// Initialize with the specified box boundaries
	UnifBoxDist(std::vector<std::array<double,2>> boundaries)
		: boundaries(boundaries) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The box boundaries.
	std::vector<std::array<double,2>> boundaries;
};

/// Generate points uniformly from a ball.
struct BallDist : PointGenerator {
	virtual ~BallDist() = default;

	/// Initialize with the center and radius.
	BallDist(Eigen::VectorXd center, double radius)
		: center(center), radius(radius) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The ball center.
	Eigen::VectorXd center;

	/// The ball radius.
	double radius;
};

/// Generate points uniformly from a sphere (the boundary of a ball).
struct SphereDist : PointGenerator {
	virtual ~SphereDist() = default;

	/// Initialize with the center and radius.
	SphereDist(Eigen::VectorXd center, double radius)
		: center(center), radius(radius) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The sphere center.
	Eigen::VectorXd center;

	/// The sphere radius.
	double radius;
};

/// Generate points uniformly from a simplex (d-dimensional triangle).
///
/// Algorithm based on:
/// http://www.mathworks.com/matlabcentral/newsreader/view_thread/170480
struct SimplexDist : PointGenerator {
	virtual ~SimplexDist() = default;

	/// Initialize to the unit simplex in nDim dimensions.
	SimplexDist(size_t nDim);

	/// Initialize to a simplex with the given vertices. We assume the vertices
	/// you provide are affinely-independent.
	/// Each row of vertices is a vertex position.
	SimplexDist(Eigen::MatrixXd vertices) : vertices(vertices) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The simplex vertices.
	Eigen::MatrixXd vertices;
};

/// Generate points from a Multivariate Normal distribution.
struct MVNDist : PointGenerator {
	virtual ~MVNDist() = default;

	/// Initialize using the specified mean and covariance.
	MVNDist(Eigen::VectorXd mu, Eigen::MatrixXd cov) : mu(mu), cov(cov) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The mean.
	Eigen::VectorXd mu;

	/// The covariance matrix.
	Eigen::MatrixXd cov;
};

/// Generate points from a mixture model.
struct MixtureModel : PointGenerator {
	virtual ~MixtureModel() = default;

	/// Initialize using the specified components and mixture probabilites.
	MixtureModel(std::vector<std::shared_ptr<PointGenerator>> components,
		Eigen::VectorXd mixture)
		: components(components), mixture(mixture) {}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The mixture components.
	std::vector<std::shared_ptr<PointGenerator>> components;

	/// The mixture probaiblities.
	Eigen::VectorXd mixture;
};

/// Generate points from a Swiss Roll distribution.
/// This is a 2D manifold in the shape of a spiral inside a 3D background space.
struct SwissRollGenerator : PointGenerator {
	virtual ~SwissRollGenerator() = default;

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;
};

/// Generate points from a (possibly) scale-free process.
/// This process tends to create dense regions and sparse regions, where new
/// points are more likely to be added to already-dense regions.
/// We generate points by first deciding whether to randomly add a new point
/// uniformly within the bounding box of the set, with probability lambda.
/// The bounding box is initially the cube with range [-1, +1] in each
/// dimension. It is expanded as needed to contain all generated points.
///
/// If not, then (with probability 1-lambda) we select an existing "reference"
/// point uniformly at random and add a point close to the reference point. The
/// new point's position is chosen one axis at a time by selecting a random sign
/// and then drawing an offset from the reference point's position from an
/// exponential distribution with mean mu.
struct ScaleFreeGenerator : PointGenerator {
	virtual ~ScaleFreeGenerator() = default;

	/// Construct with exploration rate lambda and expected distance mu.
	ScaleFreeGenerator(size_t nDim, double lambda, double mu)
		: boundingBox(std::vector<std::array<double,2>>(nDim, {-1.0, +1.0}))
		, lambda(lambda)
		, mu(mu)
	{}

	/// Generate a batch of points.
	Eigen::MatrixXd generate(size_t nPoints) override;

	/// The distribution over the bounding box. The box will be expanded as
	/// needed when points are drawn outside its bounds.
	UnifBoxDist boundingBox;

	/// The probability of generating a point at random over the bounding box.
	double lambda;

	/// The average distance from a reference point along each axis.
	double mu;
};

} // end namespace util
} // end namespace OGT_NAMESPACE
#endif /* OGT_UTIL_DISTS_HPP */
