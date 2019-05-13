/// @file  embed_cmp_with_soe.cpp
/// @brief Implements Soft Ordinal Embedding.

#include <ogt/embed/cmp.hpp>
#include <ogt/embed/dlib_opt.hpp>
#include <ogt/linalg/linalg.hpp>
#include <chrono>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using OGT_NAMESPACE::linalg::dist;
using std::max;
using std::min;
using std::vector;

namespace OGT_NAMESPACE {
namespace embed {

/// Internal method to assign unit weights to all triples. Equivalent to not
/// using weights.
double unitWeights(size_t) { return 1.0; }

/// The data container for the SOE objective information
template<typename Iter>
struct SOEBase {
	SOEBase(Iter consBegin, Iter consEnd, std::function<double(size_t)> weights,
		double scale)
		: consBegin(consBegin)
		, consEnd(consEnd)
		, weights(weights)
		, scale(scale)
		, started(std::chrono::steady_clock::now())
	{
	}
	const Iter consBegin;
	const Iter consEnd;
	const std::function<double(size_t)> weights;
	const double scale;
	const std::chrono::steady_clock::time_point started;
};

/// The SOE optimization objective
template<typename Iter>
struct SOEObjective : SOEBase<Iter> {

	/// Constructor.
	SOEObjective(Iter consBegin, Iter consEnd,
		std::function<double(size_t)> weights, double scale)
		: SOEBase<Iter>(consBegin, consEnd, weights, scale) {}

	/// Calculate loss
	double operator()(const MatrixXd& X) const {
		double stress = 0;
		size_t idx = 0;
		size_t active = 0;
		for (auto it = this->consBegin; it != this->consEnd; ++it) {
			auto con = *it;
			double dab = dist(X, con.a, con.b);
			double dcd = dist(X, con.c, con.d);
			if(dab + this->scale > dcd) {
				double val = dab + this->scale - dcd;
				stress += this->weights(idx) * val * val;
				active++;
			}
			idx++;
		}
		std::chrono::duration<double> diffTime = std::chrono::steady_clock::now() - this->started;
		std::cerr << "SOE objective: " << stress
			<< " active triplets: " << active
			<< " time: " << diffTime.count()
			<< std::endl;
		return stress;
	}
};

/// The gradient of the SOE optimization objective
template<typename Iter>
struct SOEGradient : SOEBase<Iter> {

	/// Constructor.
	SOEGradient(Iter consBegin, Iter consEnd,
		std::function<double(size_t)> weights, double scale)
		: SOEBase<Iter>(consBegin, consEnd, weights, scale) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& X, MatrixXd& grad) const {
		const size_t nDim = X.cols();
		size_t idx = 0;
		for (auto it = this->consBegin; it != this->consEnd; ++it) {
			auto con = *it;
			const auto& xa = X.row(con.a);
			const auto& xb = X.row(con.b);
			const auto& xc = X.row(con.c);
			const auto& xd = X.row(con.d);
			double dab = (xa - xb).norm();
			double dcd = (xc - xd).norm();
			double dab_denom = max(dab, 1e-5);
			double dcd_denom = max(dcd, 1e-5);
			double gamma = 2 * this->weights(idx) * (dab + this->scale - dcd);
			if (dab + this->scale > dcd) {
				if (con.a != con.c && con.a != con.d) {
					for (size_t s = 0; s < nDim; s++) {
						grad(con.a,s) += gamma * ((xa(s) - xb(s)) / dab_denom);
						if (con.b != con.c && con.b != con.d) {
							grad(con.b,s) += gamma * ((xb(s) - xa(s)) / dab_denom);
							grad(con.c,s) += gamma * ((xd(s) - xc(s)) / dcd_denom);
							grad(con.d,s) += gamma * ((xc(s) - xd(s)) / dcd_denom);
						} else if (con.b == con.c) {
							grad(con.b,s) += gamma * ((xb(s) - xa(s)) / dab_denom
								- (xb(s) - xd(s)) / dcd_denom);
							grad(con.d,s) += gamma * ((xc(s) - xd(s)) / dcd_denom);
						} else if (con.b == con.d) {
							grad(con.b,s) += gamma * ((xb(s) - xa(s)) / dab_denom
								- (xb(s) - xc(s)) / dcd_denom);
							grad(con.c,s) += gamma * ((xd(s) - xc(s)) / dcd_denom);
						}
					}
				} else if (con.a == con.c) {
					for(size_t s = 0; s < nDim;s++){
						grad(con.a,s) += gamma * ((xa(s) - xb(s)) / dab_denom
							- (xa(s) - xd(s)) / dcd_denom);
						grad(con.b,s) += gamma * ((xb(s) - xa(s)) / dab_denom);
						grad(con.d,s) += gamma * ((xc(s) - xd(s)) / dcd_denom);
					}
				} else if (con.a == con.d) {
					for(size_t s = 0; s < nDim;s++){
						grad(con.a,s) += gamma * ((xa(s) - xb(s)) / dab_denom
							- (xa(s) - xc(s)) / dcd_denom);
						grad(con.b,s) += gamma * ((xb(s) - xa(s)) / dab_denom);
						grad(con.c,s) += gamma * ((xd(s) - xc(s)) / dcd_denom);
					}
				}
			}
			idx++;
		}
	}
};

/// Internal version, which takes iterators.
template<typename Iter>
EmbedResult embedCmpWithSOE(Iter consBegin, Iter consEnd,
	std::function<double(size_t)> weights, const MatrixXd& X0,
	std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> adjustment,
	EmbedConfig config) {
	const double scale = config.margin ? config.margin
		: fitSOEScale(consBegin, consEnd, X0);
	DlibResult result;
	if (adjustment) {
		result = optimizeMatrixWithAdjustment(X0,
			SOEObjective<Iter>(consBegin, consEnd, weights, scale),
			SOEGradient<Iter>(consBegin, consEnd, weights, scale),
			adjustment,
			config.minDelta,
			config.maxIter,
			config.verbose);
	} else {
		result = optimizeMatrix(X0,
			SOEObjective<Iter>(consBegin, consEnd, weights, scale),
			SOEGradient<Iter>(consBegin, consEnd, weights, scale),
			config.minDelta,
			config.maxIter,
			config.verbose);
	}
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}

/// Embeds a dataset using Soft Ordinal Embedding.
EmbedResult embedCmpWithSOE(vector<CmpConstraint> cons, const MatrixXd& X0,
	EmbedConfig config) {
	return embedCmpWithSOE(cons.begin(), cons.end(), unitWeights,
		X0, nullptr, config);
}

/// Embeds using a weighted variant of Soft Ordinal Embedding. The loss function
/// is modified so that each constraint has a corresponding weight which is
/// multiplied to the loss incurred when that constraint is violated.
/// It is assumed, but not verified, that the weights are non-negative and sum
/// to one.
EmbedResult embedCmpWithSOEWeighted(vector<CmpConstraint> cons,
	const VectorXd &weights, const MatrixXd &X0, EmbedConfig config) {
	return embedCmpWithSOE(cons.begin(), cons.end(), weights, X0, nullptr, config);
}

/// An iterator which emits triples based on the k-nearest neighbors.
struct KnnIter {

	/// Constructor.
	KnnIter(const MatrixXi& knn) : knn(knn), i(0), j(0), k(0) {}

	/// Get an iterator past the end
	KnnIter end() const {
		KnnIter it(knn);
		it.i = knn.rows();
		return it;
	}

	/// Inequality check.
	bool operator!=(const KnnIter& other) const {
		return i != other.i || j != other.j || k != other.k;
	}

	/// Increment operator.
	KnnIter& operator++() {
		if (i < knn.rows()) {

			// Increment over non-nearest neighbors
			k = (k + 1) % knn.rows();
			if (k == 0) {

				// Increment over nearest neighbors
				j = (j + 1) % knn.cols();
				if (j == 0) {

					// Increment over "head" objects
					++i;
				}
			}
		}
		return *this;
	}

	/// Dereference operator.
	CmpConstraint operator*() {
		return CmpConstraint(i, knn(i,j), k);
	}

	const MatrixXi& knn;
	Index i, j, k;
};

/// Embeds a dataset using Local Ordinal Embedding.
EmbedResult embedKnnWithLOE(const MatrixXi& knn, const MatrixXd& X0,
	EmbedConfig config) {
	KnnIter consBegin(knn);
	return embedCmpWithSOE(consBegin, consBegin.end(), unitWeights, X0, nullptr, config);
}

/// Internal version, which takes iterators.
template<typename Iter>
double fitSOEScale(Iter consBegin, Iter consEnd, const MatrixXd& X) {
	double minScale = 0.1;
	for (auto it = consBegin; it != consEnd; ++it) {
		auto con = *it;
		double dab = dist(X, con.a, con.b);
		double dcd = dist(X, con.c, con.d);
		if (dab < dcd) {
			minScale = min(minScale, dcd - dab);
		}
	}
	return minScale;
}

/// Pick an optimal SOE scale parameter.
double fitSOEScale(vector<CmpConstraint> cons, const MatrixXd& X) {
	return fitSOEScale(cons.begin(), cons.end(), X);
}

/// The base class for single-point embedding.
struct SOEPtBase {
	SOEPtBase(const std::vector<size_t>& rank, const MatrixXd& X, double scale)
		: rank(rank), X(X), scale(scale) {}

	const std::vector<size_t>& rank;
	const MatrixXd& X;
	const double scale;
};

/// The SOE optimization objective for a single point
struct SOEPtObjective : SOEPtBase {

	/// Constructor.
	SOEPtObjective(const std::vector<size_t>& rank, const MatrixXd& X,
		double scale)
		: SOEPtBase(rank, X, scale) {}

	/// Calculate loss
	double operator()(const MatrixXd& pos) const {
		double stress = 0;
		for (size_t idxc = 1; idxc < rank.size(); idxc++) {
			for (size_t idxb = 0; idxb < idxc; idxb++) {
				const size_t b = rank[idxb], c = rank[idxc];
				const VectorXd xa = pos;
				const VectorXd xb = X.row(b);
				const VectorXd xc = X.row(c);
				double dab = (xa - xb).norm();
				double dac = (xa - xc).norm();
				if(dab + this->scale > dac) {
					double val = dab + this->scale - dac;
					stress += val * val;
				}
			}
		}
		return stress;
	}
};

/// The gradient of the SOE optimization objective
struct SOEPtGradient : SOEPtBase {

	/// Constructor.
	SOEPtGradient(const std::vector<size_t>& rank, const MatrixXd& X,
		double scale)
		: SOEPtBase(rank, X, scale) {}

	/// Calculate the gradient.
	void operator()(const MatrixXd& pos, MatrixXd& grad) const {
		const size_t nDim = X.cols();
		for (size_t idxc = 1; idxc < rank.size(); idxc++) {
			for (size_t idxb = 0; idxb < idxc; idxb++) {
				const size_t b = rank[idxb], c = rank[idxc];
				const VectorXd xa = pos;
				const VectorXd xb = X.row(b);
				const VectorXd xc = X.row(c);
				double dab = (xa - xb).norm();
				double dac = (xa - xc).norm();
				double dab_denom = max(dab, 1e-5);
				double dcd_denom = max(dac, 1e-5);
				double gamma = 2 * (dab + this->scale - dac);
				if (dab + this->scale > dac) {
					for(size_t s = 0; s < nDim; s++){
						grad(s) += gamma * ((xa(s) - xb(s)) / dab_denom
							- (xa(s) - xc(s)) / dcd_denom);
					}
				}
			}
		}
	}
};

/// Embeds a point to satisfy a ranking of some set of fixed points, as much
/// as possible.
EmbedResult embedPtRankingWithSOE(const vector<size_t> &rank, const MatrixXd& X,
	const Eigen::VectorXd& pos0, EmbedConfig config) {
	auto result = optimizeMatrix(pos0,
		SOEPtObjective(rank, X, config.margin),
		SOEPtGradient(rank, X, config.margin),
		config.minDelta,
		config.maxIter,
		config.verbose);
	EmbedResult er;
	er.X = result.answer;
	er.loss = result.objective;
	return er;
}


} // end namespace embed
} // end namespace OGT_NAMESPACE
