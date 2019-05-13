/// @file  geometric.hpp
/// @brief Declarations for geometric embedding methods

#pragma once
#ifndef OGT_EMBED_GEOMETRIC_HPP
#define OGT_EMBED_GEOMETRIC_HPP

#include <ogt/config.hpp>
#include <ogt/core/collection.hpp>
#include <ogt/core/traversal.hpp>
#include <Eigen/Dense>
#include <memory>

namespace OGT_NAMESPACE {
namespace embed {

/// The result of a geometric embedding method
struct GeometricEmbedding {

	/// The embedding
	Eigen::MatrixXd X;
};

/// An abstract base class for geometry-based embedding methods
struct GeometricEmbedder {

	/// Random landmark selection
	static const std::string LM_RANDOM;

	/// Farthest-rank-first landmark selection
	static const std::string LM_FRFT;

	/// Most-orthogonal landmark selection
	static const std::string LM_ORTH;

	/// Random distance bounds
	static const std::string BND_RANDOM;

	/// True distance bounds
	static const std::string BND_TRUE;

	/// Circle radius distance bounds
	static const std::string BND_CIRCLES;

	/// Noisy distance bounds
	static const std::string BND_NOISY;

	/// Create a new instance of a landmark embedder
	static std::shared_ptr<GeometricEmbedder> CreateLM(
		std::shared_ptr<OGT_NAMESPACE::core::Collection> coll,
		std::string lmOrder, std::string distBounds, size_t nDim,
		double& landmarkTime, double& boundsTime, double& embeddingTime);

	/// Virtual destructor.
	virtual ~GeometricEmbedder() = default;

	/// Get any initialization error (or an empty string if none)
	virtual std::string error() const = 0;

	/// Run the embedding algorithm.
	virtual GeometricEmbedding embed() = 0;

	/// Get an embedding based on what's currently known.
	virtual GeometricEmbedding embedCurrent() const = 0;

	/// Get the collection we are embedding.
	virtual std::shared_ptr<OGT_NAMESPACE::core::Collection> coll() = 0;

	/// Set distance bounds for a landmark.
	virtual void setBounds(size_t landmark, Eigen::VectorXd& lb,
		Eigen::VectorXd& ub) = 0;
};

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_GEOMETRIC_HPP */
