/// @file  dlib_opt.cpp
/// @brief Implements helpers for dlib optimizers.

#include <ogt/embed/dlib_opt.hpp>
#include <iostream>

using dlib::find_min;
using dlib::lbfgs_search_strategy;
using dlib::objective_delta_stop_strategy;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OGT_NAMESPACE {
namespace embed {

/// Converts an Eigen vector to a dlib vector.
dlib_vector eigen_to_dlib_vector(const VectorXd& in) {
	dlib_vector out(in.size());
	for (Index j = 0; j < in.size(); j++) {
		out(j) = in(j);
	}
	return out;
}

/// Converts an Eigen matrix to a dlib matrix.
dlib_matrix eigen_to_dlib_matrix(const MatrixXd& in) {
	dlib_matrix out(in.rows(), in.cols());
	for (Index i = 0; i < in.rows(); i++) {
		for (Index j = 0; j < in.cols(); j++) {
			out(i,j) = in(i,j);
		}
	}
	return out;
}

/// Converts an Eigen matrix to a dlib vector.
dlib_vector eigen_matrix_to_dlib_vector(const MatrixXd& in) {
	dlib_vector out(in.size());
	for (Index i = 0; i < in.rows(); i++) {
		for (Index j = 0; j < in.cols(); j++) {
			out(i * in.cols() + j) = in(i,j);
		}
	}
	return out;
}

/// Converts a dlib vector to an Eigen vector.
VectorXd dlib_to_eigen_vector(const dlib_vector& in) {
	VectorXd out(in.size());
	for (long j = 0; j < in.size(); j++) {
		out(j) = in(j);
	}
	return out;
}

/// Converts a dlib matrix to an Eigen matrix.
MatrixXd dlib_to_eigen_matrix(const dlib_matrix& in) {
	MatrixXd out(in.nr(), in.nc());
	for (long i = 0; i < in.nr(); i++) {
		for (long j = 0; j < in.nc(); j++) {
			out(i,j) = in(i,j);
		}
	}
	return out;
}

/// Converts a dlib vector to an Eigen matrix.
MatrixXd dlib_vector_to_eigen_matrix(const dlib_vector& in, Index rows,
	Index cols) {
	assert(in.size() == rows * cols);
	MatrixXd out(rows, cols);
	for (Index i = 0; i < rows; i++) {
		for (Index j = 0; j < cols; j++) {
			out(i,j) = in((i * cols + j));
		}
	}
	return out;
}

/// Optimize a matrix of parameters using L-BFGS.
DlibResult optimizeMatrix(MatrixXd m0,
	std::function<double(const MatrixXd&)> objective,
	std::function<void(const MatrixXd&,MatrixXd&)> gradient,
	double minDelta, size_t maxIter, bool verbose) {
	const Index rows = m0.rows(), cols = m0.cols();
	double pre = objective(m0);
	objective_delta_stop_strategy stopWhen(minDelta, maxIter);
	if (verbose) {
		std::cout << "Initial objective: " << pre << std::endl;
		stopWhen = stopWhen.be_verbose();
	}
	dlib_vector params = eigen_matrix_to_dlib_vector(m0);

	DlibResult result;
	result.objective = find_min(
		lbfgs_search_strategy(1000),
		stopWhen,
		[&](const dlib_vector& params) -> double {
			return objective(dlib_vector_to_eigen_matrix(params, rows, cols));
		},
		[&](const dlib_vector& params) -> dlib_vector {
			MatrixXd grad = MatrixXd::Zero(rows, cols);
			gradient(dlib_vector_to_eigen_matrix(params, rows, cols), grad);
			return eigen_matrix_to_dlib_vector(grad);
		},
		params,
		-1);
	result.answer = dlib_vector_to_eigen_matrix(params, m0.rows(), m0.cols());
	if (verbose) {
		std::cout << "Final objective: " << result.objective << " delta: "
			<< pre - result.objective << std::endl;
	}
	return result;
}

/// find_min() as copied from dlib, but with an extra adjustment step
template <
typename search_strategy_type,
typename stop_strategy_type,
typename funct, 
typename funct_der, 
typename funct_adj, 
typename T
>
double find_min_with_adjustment (
	search_strategy_type search_strategy,
	stop_strategy_type stop_strategy,
	const funct& f, 
	const funct_der& der, 
	const funct_adj& adj,
	T& x, 
	double min_f
	)
{
	COMPILE_TIME_ASSERT(dlib::is_matrix<T>::value);
	// The starting point (i.e. x) must be a column vector.  
	COMPILE_TIME_ASSERT(T::NC <= 1);

	DLIB_CASSERT (
		is_col_vector(x),
		"\tdouble find_min()"
		<< "\n\tYou have to supply column vectors to this function"
		<< "\n\tx.nc():    " << x.nc()
		);


	T g, s;

	double f_value = f(x);
	g = der(x);

	if (!dlib::is_finite(f_value))
		throw dlib::error("The objective function generated non-finite outputs");
	if (!dlib::is_finite(g))
		throw dlib::error("The objective function generated non-finite outputs");

	while(stop_strategy.should_continue_search(x, f_value, g) && f_value > min_f)
	{
		s = search_strategy.get_next_direction(x, f_value, g);

		double alpha = line_search(
			make_line_search_function(f,x,s, f_value),
			f_value,
			make_line_search_function(der,x,s, g),
			dot(g,s), // compute initial gradient for the line search
			search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f,
			search_strategy.get_max_line_search_iterations());

		// Take the search step indicated by the above line search
		x += alpha*s;

		// Adjust the results
		x = adj(x);

		if (!dlib::is_finite(f_value))
			throw dlib::error("The objective function generated non-finite outputs");
		if (!is_finite(g))
			throw dlib::error("The objective function generated non-finite outputs");
	}

	return f_value;
}

/// find_min() as copied from dlib, but keeping the configuration that obtains
/// the minimum objective value.
template <
typename search_strategy_type,
typename stop_strategy_type,
typename funct, 
typename funct_der, 
typename T
>
double find_min_keep_min (
	search_strategy_type search_strategy,
	stop_strategy_type stop_strategy,
	const funct& f, 
	const funct_der& der, 
	T& x, 
	double min_f
	)
{
	COMPILE_TIME_ASSERT(dlib::is_matrix<T>::value);
	// The starting point (i.e. x) must be a column vector.  
	COMPILE_TIME_ASSERT(T::NC <= 1);

	DLIB_CASSERT (
		is_col_vector(x),
		"\tdouble find_min()"
		<< "\n\tYou have to supply column vectors to this function"
		<< "\n\tx.nc():    " << x.nc()
		);


	T g, s;

	double f_value = f(x);
	g = der(x);
	double min_f_found = f_value;
	T best_x = x;

	if (!dlib::is_finite(f_value))
		throw dlib::error("The objective function generated non-finite outputs");
	if (!dlib::is_finite(g))
		throw dlib::error("The objective function generated non-finite outputs");

	while(stop_strategy.should_continue_search(x, f_value, g) && f_value > min_f)
	{
		s = search_strategy.get_next_direction(x, f_value, g);

		double alpha = line_search(
			make_line_search_function(f,x,s, f_value),
			f_value,
			make_line_search_function(der,x,s, g),
			dot(g,s), // compute initial gradient for the line search
			search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f,
			search_strategy.get_max_line_search_iterations());

		// Take the search step indicated by the above line search
		x += alpha*s;

		if (!dlib::is_finite(f_value))
			throw dlib::error("The objective function generated non-finite outputs");
		if (!is_finite(g))
			throw dlib::error("The objective function generated non-finite outputs");

		// Keep the best result
		if (f_value < min_f_found) {
			min_f_found = f_value;
			best_x = x;
		}
	}

	// Return the best result
	x = best_x;
	f_value = min_f_found;

	return f_value;
}

/// Optimize a matrix of parameters using L-BFGS, but make some adjustment in
/// between optimization steps.
DlibResult optimizeMatrixWithAdjustment(Eigen::MatrixXd m0,
	std::function<double(const Eigen::MatrixXd&)> objective,
	std::function<void(const Eigen::MatrixXd&,Eigen::MatrixXd&)> gradient,
	std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> adjustment,
	double minDelta, size_t maxIter, bool verbose) {
	const Index rows = m0.rows(), cols = m0.cols();
	double pre = objective(m0);
	objective_delta_stop_strategy stopWhen(minDelta, maxIter);
	if (verbose) {
		std::cout << "Initial objective: " << pre << std::endl;
		stopWhen = stopWhen.be_verbose();
	}
	dlib_vector params = eigen_matrix_to_dlib_vector(m0);

	DlibResult result;
	result.objective = find_min_with_adjustment(
		lbfgs_search_strategy(10),
		stopWhen,
		[&](const dlib_vector& params) -> double {
			return objective(dlib_vector_to_eigen_matrix(params, rows, cols));
		},
		[&](const dlib_vector& params) -> dlib_vector {
			MatrixXd grad = MatrixXd::Zero(rows, cols);
			gradient(dlib_vector_to_eigen_matrix(params, rows, cols), grad);
			return eigen_matrix_to_dlib_vector(grad);
		},
		[&](const dlib_vector& params) -> dlib_vector {
			return eigen_matrix_to_dlib_vector(adjustment(
				dlib_vector_to_eigen_matrix(params, rows, cols)));
		},
		params,
		-1);
	result.answer = dlib_vector_to_eigen_matrix(params, m0.rows(), m0.cols());
	if (verbose) {
		std::cout << "Final objective: " << result.objective << " delta: "
			<< pre - result.objective << std::endl;
	}
	return result;
}

/// Optimize a matrix of parameters using L-BFGS, but keep the configuration
/// we find with the minimum objective score.
DlibResult optimizeMatrixKeepMin(Eigen::MatrixXd m0,
	std::function<double(const Eigen::MatrixXd&)> objective,
	std::function<void(const Eigen::MatrixXd&,Eigen::MatrixXd&)> gradient,
	double minDelta, size_t maxIter, bool verbose) {
	const Index rows = m0.rows(), cols = m0.cols();
	double pre = objective(m0);
	objective_delta_stop_strategy stopWhen(minDelta, maxIter);
	if (verbose) {
		std::cout << "Initial objective: " << pre << std::endl;
		stopWhen = stopWhen.be_verbose();
	}
	dlib_vector params = eigen_matrix_to_dlib_vector(m0);

	DlibResult result;
	result.objective = find_min_keep_min(
		lbfgs_search_strategy(10),
		stopWhen,
		[&](const dlib_vector& params) -> double {
			return objective(dlib_vector_to_eigen_matrix(params, rows, cols));
		},
		[&](const dlib_vector& params) -> dlib_vector {
			MatrixXd grad = MatrixXd::Zero(rows, cols);
			gradient(dlib_vector_to_eigen_matrix(params, rows, cols), grad);
			return eigen_matrix_to_dlib_vector(grad);
		},
		params,
		-1);
	result.answer = dlib_vector_to_eigen_matrix(params, m0.rows(), m0.cols());
	if (verbose) {
		std::cout << "Final objective: " << result.objective << " delta: "
			<< pre - result.objective << std::endl;
	}
	return result;
}

} // end namespace embed
} // end namespace OGT_NAMESPACE
