/// @file  dlib_opt.hpp
/// @brief Includes dlib and declares various helpers.

#pragma once
#ifndef OGT_EMBED_DLIB_OPT_HPP
#define OGT_EMBED_DLIB_OPT_HPP

#include <ogt/config.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <dlib/optimization.h>
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <functional>

namespace OGT_NAMESPACE {
namespace embed {

/// A vector as used by dlib optimizers.
typedef dlib::matrix<double,0,1> dlib_vector;

/// A matrix as used by dlib optimizers.
typedef dlib::matrix<double> dlib_matrix;

/// Converts an Eigen vector to a dlib vector.
dlib_vector eigen_to_dlib_vector(const Eigen::VectorXd& in);

/// Converts an Eigen matrix to a dlib matrix.
dlib_matrix eigen_to_dlib_matrix(const Eigen::MatrixXd& in);

/// Converts an Eigen matrix to a dlib vector.
dlib_vector eigen_matrix_to_dlib_vector(const Eigen::MatrixXd& in);

/// Converts a dlib vector to an Eigen vector.
Eigen::VectorXd dlib_to_eigen_vector(const dlib_vector& in);

/// Converts a dlib vector to an Eigen matrix.
Eigen::MatrixXd dlib_vector_to_eigen_matrix(const dlib_vector& in,
	Eigen::Index rows, Eigen::Index cols);

/// Converts a dlib matrix to an Eigen matrix.
Eigen::MatrixXd dlib_to_eigen_matrix(const dlib_matrix& in);

/// The result of a generic optimization run.
struct DlibResult {

	/// The result discovered.
	Eigen::MatrixXd answer;

	/// The value of the objective at termination.
	double objective;
};

/// Optimize a matrix of parameters using L-BFGS.
///
/// Note that dlib guarantees that objective will always be called just before
/// gradient, so gradient can safely use information computed by objective.
DlibResult optimizeMatrix(Eigen::MatrixXd m0,
	std::function<double(const Eigen::MatrixXd&)> objective,
	std::function<void(const Eigen::MatrixXd&,Eigen::MatrixXd&)> gradient,
	double minDelta, size_t maxIter, bool verbose);

/// Optimize a matrix of parameters using L-BFGS, but make some adjustment in
/// between optimization steps. This is useful for projecting a kernel back onto
/// the PSD cone.
///
/// Note that dlib guarantees that objective will always be called just before
/// gradient, so gradient can safely use information computed by objective.
DlibResult optimizeMatrixWithAdjustment(Eigen::MatrixXd m0,
	std::function<double(const Eigen::MatrixXd&)> objective,
	std::function<void(const Eigen::MatrixXd&,Eigen::MatrixXd&)> gradient,
	std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> adjustment,
	double minDelta, size_t maxIter, bool verbose);

/// Optimize a matrix of parameters using L-BFGS, but keep the configuration
/// we find with the minimum objective score.
DlibResult optimizeMatrixKeepMin(Eigen::MatrixXd m0,
	std::function<double(const Eigen::MatrixXd&)> objective,
	std::function<void(const Eigen::MatrixXd&,Eigen::MatrixXd&)> gradient,
	double minDelta, size_t maxIter, bool verbose);

} // end namespace embed
} // end namespace OGT_NAMESPACE
#endif /* OGT_EMBED_DLIB_OPT_HPP */
