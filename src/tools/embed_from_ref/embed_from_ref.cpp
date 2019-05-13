/// @file  embed_from_ref.cpp
/// @brief Embed from an embedded subset

#include <boost/program_options.hpp>
#include <ogt/ogt.hpp>
#include <future>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace boost::program_options;
using dlib::find_min;
using dlib::newton_search_strategy;
using dlib::bfgs_search_strategy;
using dlib::objective_delta_stop_strategy;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using ogt::core::AB_LT_AC;
using ogt::core::Collection;
using ogt::core::createCosinePosOracle;
using ogt::core::createEuclideanPosOracle;
using ogt::core::createJaccardPosOracle;
using ogt::core::Oracle;
using ogt::embed::dlib_matrix;
using ogt::embed::dlib_vector;
using ogt::embed::dlib_to_eigen_matrix;
using ogt::embed::eigen_to_dlib_matrix;
using ogt::embed::dlib_to_eigen_vector;
using ogt::embed::eigen_to_dlib_vector;
using ogt::embed::CmpConstraint;
using ogt::embed::embedCmpWithSOE;
using ogt::embed::EmbedConfig;
using ogt::embed::EmbedErr;
using ogt::embed::EmbedResult;
using ogt::embed::sphere_intersection_margin;
using ogt::io::isSparseFile;
using ogt::io::loadDenseMatrix;
using ogt::io::loadSparseMatrix;
using ogt::io::openSink;
using ogt::io::saveEmbedding;
using ogt::linalg::DenseNZIterator;
using ogt::linalg::posToDist;
using ogt::util::random_nchoosek;
using ogt::util::random_subset;
using std::async;
using std::cerr;
using std::cout;
using std::endl;
using std::future;
using std::make_shared;
using std::map;
using std::min;
using std::set;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

std::ostream& log() {
	std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    cerr << std::put_time(&tm, "%c %Z") << ' ';
    return cerr;
}

const char * DIST_EUCLIDEAN = "EUCLIDEAN";
const char * DIST_COSINE = "COSINE";
const char * DIST_JACCARD = "JACCARD";

struct pt_pos {
	pt_pos(size_t pt, VectorXd pos, vector<CmpConstraint> cmps)
		: pt(pt), pos(pos), cmps(cmps), createdAt(clock()) {}
	const size_t pt;
	const VectorXd pos;
	const vector<CmpConstraint> cmps;
	const clock_t createdAt;
};

struct idx_dist {
	idx_dist(size_t idx, double dist) : idx(idx), dist(dist) {}
	size_t idx;
	double dist;
};

struct fft_idx {
	fft_idx(size_t idx, double dist) : idx(idx), dist(dist) {}

	void findDists(const vector<size_t>& subset, const MatrixXd& refD) {
		// Prepare the sorted distance vector
		dists.clear();
		dists.reserve(refD.rows());
		for (Index i = 0; i < refD.cols(); i++) {
			dists.emplace_back(subset[i], refD(idx, i));
		}
		std::sort(dists.begin(), dists.end(), [](const auto& a, const auto& b) {
			return a.dist < b.dist;
		});
	}

	size_t idx;
	double dist;
	vector<idx_dist> dists;
};

struct pt_shell {
	pt_shell() : center(0), minDist(0), maxDist(0) {}
	size_t center;
	double minDist;
	double maxDist;
};

/// Find a farthest-first traversal of a set
vector<fft_idx> fft(const vector<size_t>& subset, const MatrixXd& refD) {
	vector<fft_idx> order;
	size_t nObj = refD.rows();
	order.reserve(nObj);

	// Find the first point, on the convex hull
	VectorXd dist = refD.row(0);
	fft_idx choice(0, 0);
	for (size_t i = 1; i < nObj; i++) {
		if (dist(i) > choice.dist) {
			choice.idx = i;
			choice.dist = dist(i);
		}
	}
	choice.findDists(subset, refD);
	order.push_back(choice);
	vector<fft_idx> minDist;
	for (size_t i = 0; i < nObj; i++) {
		if (i != choice.idx) {
			minDist.emplace_back(i, refD(choice.idx, i));
		}
	}

	// Iterate through the other points until done
	while (!minDist.empty()) {
		std::sort(minDist.begin(), minDist.end(), [](const auto& a, const auto& b) {
			return a.dist < b.dist;
		});
		choice = minDist.back();
		choice.findDists(subset, refD);
		minDist.pop_back();
		order.push_back(choice);
		for (auto& ptd : minDist) {
			ptd.dist = std::min(ptd.dist, refD(choice.idx, ptd.idx));
		}
	}

	return order;
}


/// Tries to use the suggested anchor and pick distance bounds to the point.
bool find_anchor(size_t pt, const fft_idx& ptd,
	const vector<size_t>& subset, shared_ptr<Oracle> oracle,
	pt_shell& anchor, vector<CmpConstraint>& cmps) {
	anchor.center = ptd.idx;
	size_t center = subset[ptd.idx];

	// Make sure the point can be compared to the proposed anchor
	if ((*oracle)(center, ptd.dists.front().idx, pt) != AB_LT_AC ||
		(*oracle)(center, pt, ptd.dists.back().idx) != AB_LT_AC) {
		return false;
	}

	// Find the point in the vector
	auto ref = ptd.dists.begin();
	auto pre = ref;
	for (; (*oracle)(center, ref->idx, pt) == AB_LT_AC; ref = next(ref)) {
		// cmps.emplace_back(center, ref->idx, pt);
		pre = ref;
		anchor.minDist = ref->dist;
	}
	for (; ref != ptd.dists.end() 
		&& (*oracle)(center, pt, ref->idx) != AB_LT_AC; ref = next(ref)) {}
	if (ref == ptd.dists.end()) {
		return false;
	}
	anchor.maxDist = ref->dist;
	cmps.emplace_back(center, pre->idx, pt);
	cmps.emplace_back(center, pt, ref->idx);
	// auto post = std::lower_bound(ptd.dists.begin(), ptd.dists.end(), pt,
	// 	[&](const idx_dist& ref, size_t pt) {
	// 		return (*oracle)(center, ref.idx, pt) == AB_LT_AC;
	// 	});
	// if (post == ptd.dists.begin() || post == ptd.dists.end()) {
	// 	return false;
	// }
	// cmps.emplace_back(center, pt, ref->idx);
	// anchor.maxDist = post->dist;

	// auto pre = prev(post);
	// for (; (*oracle)(center, pre->idx, pt) != AB_LT_AC; pre = prev(pre)) {}
	// anchor.minDist = pre->dist;
	// cmps.emplace_back(center, pre->idx, pt);
	// cmps.emplace_back(center, pt, post->idx);
	return true;
}

/// The least-squares objective for sphere_intersection_opt
struct opt_objective {
	opt_objective(const MatrixXd& center, const VectorXd& radius,
		const VectorXd& margin)
		: center(center), radius(radius), margin(margin) {}
	const MatrixXd& center;
	const VectorXd& radius;
	const VectorXd& margin;

	double operator()(const dlib_vector& dy) const {
		double res = 0;
		VectorXd y = dlib_to_eigen_vector(dy);
		for (Index j = 0; j < center.cols(); j++) {
			double val = (y - center.col(j)).norm() - radius(j);
			if (fabs(val) > margin(j)) {
				res += val * val;
			}
		}
		return res;
	}
};

/// The gradient of the least-squares objective for sphere_intersection_opt
struct opt_objective_gradient {
	opt_objective_gradient(const MatrixXd& center, const VectorXd& radius,
		const VectorXd& margin)
		: center(center), radius(radius), margin(margin) {}
	const MatrixXd& center;
	const VectorXd& radius;
	const VectorXd& margin;

	dlib_vector operator()(const dlib_vector& dy) const {
		VectorXd y = dlib_to_eigen_vector(dy);
		VectorXd res = VectorXd::Zero(y.size());
		for (Index j = 0; j < center.cols(); j++) {
			VectorXd diff = y - center.col(j);
			double dist = diff.norm();
			if (fabs(dist - radius(j)) > margin(j)) {
				res += ((dist - radius(j)) / dist) * diff;
			}
		}
		return eigen_to_dlib_vector(2 * res);
	}
};

// /// The Hessian of the least-squares objective for sphere_intersection_opt
// struct opt_objective_hessian {
// 	opt_objective_hessian(const MatrixXd& center, const VectorXd& radius)
// 		: center(center), radius(radius) {}
// 	const MatrixXd& center;
// 	const VectorXd& radius;

// 	dlib_matrix operator()(const dlib_vector& dy) const {
// 		VectorXd y = dlib_to_eigen_vector(dy);
// 		MatrixXd res = MatrixXd::Zero(dy.size(), dy.size());
// 		for (Index j = 0; j < center.cols(); j++) {
// 			VectorXd diff = y - center.col(j);
// 			double dist = diff.norm();
// 			double sqdist = diff.squaredNorm();
// 			double scale = (1 - ((dist - radius(j)) / dist)) / sqdist;
// 			for (Index i = 0; i < center.rows(); i++) {
// 				double val1 = y(i) - center(i,j);
// 				for (Index k = 0; k <= i; k++) {
// 					res(i,k) += val1 * diff(k) * scale;
// 					res(k,i) = res(i,k);
// 				}
// 			}
// 		}
// 		return 2 * eigen_to_dlib_matrix(res);
// 	}
// };

/// Embed a point based on shell intersection
VectorXd shell_intersection(const Eigen::MatrixXd &center,
	const Eigen::VectorXd& radius, const Eigen::VectorXd& margin, double /*eps*/,
	double min_delta, size_t max_iter, bool verbose) {

	// Using sphere intersection plus margin
	// auto locs = sphere_intersection_margin(centers, radii, thickness,
	// 	eps, minDelta, maxIter, verbose);
	// return locs[0];

	// Direct method
	const Index nDim = center.rows();
	dlib_vector dx = eigen_to_dlib_vector(VectorXd::Random(nDim));
	objective_delta_stop_strategy stopper =
		objective_delta_stop_strategy(min_delta, max_iter);
	if (verbose) {
		stopper = stopper.be_verbose();
	}
	try {
		find_min(
			// newton_search_strategy() may produce NaN in some cases?
			// newton_search_strategy(opt_objective_hessian(center, radius)),
			bfgs_search_strategy(),
			stopper,
			opt_objective(center, radius, margin),
			opt_objective_gradient(center, radius, margin),
			dx,
			0);
	} catch (dlib::error err) {
		throw EmbedErr(err.what());
	}
	return dlib_to_eigen_vector(dx);
}

/// Run a single embedding, perhaps in a thread pool.
vector<pt_pos> embed(vector<size_t> pts, size_t nDim, size_t numAnchors,
	const MatrixXd& refX,
	const vector<size_t>& subset, const vector<fft_idx>& refFFT,
	shared_ptr<Oracle> oracle) {
	vector<pt_pos> result;
	for (size_t pt : pts) {

		// Identify 2(d+1) ref. points close to pt in FFT order
		if (numAnchors == 0) {
			numAnchors = 2 * (nDim + 1);
		}
		size_t nextAnchor = 0;
		MatrixXd centers(nDim, numAnchors);
		VectorXd radii(numAnchors);
		VectorXd thickness(numAnchors);
		vector<CmpConstraint> cmps;
		for (const auto& ptd : refFFT) {
			pt_shell anchor;
			if (find_anchor(pt, ptd, subset, oracle, anchor, cmps)) {
				centers.col(nextAnchor) = refX.row(ptd.idx);
				radii(nextAnchor) = (anchor.minDist + anchor.maxDist) / 2;
				thickness(nextAnchor) = (anchor.maxDist - anchor.minDist) / 2;

				nextAnchor++;
				if (nextAnchor >= numAnchors) break;
			}
		}

		// Now embed the point using the spherical shell intersection
		try {
			const double eps = 1e-6;
			const double minDelta = 1e-9;
			const size_t maxIter = 1000;
			const bool verbose = false;
			auto loc = shell_intersection(centers, radii, thickness, eps, minDelta,
				maxIter, verbose);
			cerr << "Done embedding " << pt << endl;
			result.emplace_back(pt, loc, cmps);
		} catch(std::invalid_argument e) {
			log() << "Could not embed " << pt << ": " << e.what() << endl;
			continue;
		} catch (ogt::embed::EmbedErr e) {
			log() << "Could not embed " << pt << ": " << e.what() << endl;
			continue;
		} catch(...) {
			log() << "Could not embed " << pt << ": unknown exception" << endl;
			continue;
		}
	}
	return result;
}

/// Run the program.
int run_embed_from_ref(string posPath, string refPath, size_t numToEmbed,
	string distType, string soeCmpPath, string embPath, string orderPath,
	size_t numAnchors, double noise) {

	// Prepare to save the output order
	const clock_t started = clock();
	auto fOrder = openSink(orderPath);
	if (!fOrder) {
		log() << "Invalid --order path: could not open " << orderPath << endl;
		return 1;
	}
	auto saveOrder = [&](const pt_pos& pt) {
		(*fOrder) << static_cast<float>(pt.createdAt - started) / CLOCKS_PER_SEC
			<< "," << pt.pt << endl;
	};

	// Load the true position matrix and create a distance oracle
	log() << "Loading position matrix" << endl;
	MatrixXd truePos;
	Eigen::SparseMatrix<double, Eigen::RowMajor> spTruePos;
	shared_ptr<Oracle> oracle;
	size_t nObj, trueDim;
	if (isSparseFile(posPath)) {
		spTruePos = loadSparseMatrix(posPath);
		if (distType == DIST_EUCLIDEAN) {
			oracle = createEuclideanPosOracle(spTruePos);
		} else if (distType == DIST_COSINE) {
			oracle = createCosinePosOracle(spTruePos);
		} else if (distType == DIST_JACCARD) {
			oracle = createJaccardPosOracle<SparseMatrix<double, Eigen::RowMajor>, 
				Eigen::SparseVector<double>,
				Eigen::SparseVector<double>::InnerIterator>(spTruePos);
		} else {
			log() << "Invalid distance function: " << distType << endl;
			return 0;
		}
		nObj = spTruePos.rows();
		trueDim = spTruePos.cols();
	} else {
		truePos = loadDenseMatrix(posPath);
		if (distType == DIST_EUCLIDEAN) {
			oracle = createEuclideanPosOracle(truePos);
		} else if (distType == DIST_COSINE) {
			oracle = createCosinePosOracle(truePos);
		} else if (distType == DIST_JACCARD) {
			oracle = createJaccardPosOracle<MatrixXd, VectorXd,
				DenseNZIterator<VectorXd>>(truePos);
		} else {
			log() << "Invalid distance function: " << distType << endl;
			return 1;
		}
		nObj = truePos.rows();
		trueDim = truePos.cols();
	}
	log() << "Position matrix is " << nObj << "x" << trueDim << endl;
	if (nObj < 1) {
		log() << "Error: empty position matrix from " << posPath << endl;
		return 1;
	}
	if (noise > 0) {
		oracle = createUniformNoisyOracle(oracle, noise);
	}

	// Load the reference embedding
	log() << "Loading reference embedding" << endl;
	VectorXd objects;
	MatrixXd refX;
	try {
		auto emb = loadDenseMatrix(refPath);
		objects = emb.col(0);
		refX = emb.rightCols(emb.cols() - 1);
	} catch(std::runtime_error e) {
		log() << e.what() << endl;
		return 1;
	}
	const size_t nDim = refX.cols();
	log() << "Reference embedding is " << refX.rows() << "x" << nDim << endl;

	log() << "Finding distances for reference embedding" << endl;
	MatrixXd refD = posToDist(refX);

	// Load the SOE comparisons
	vector<CmpConstraint> cmps;
	if (!soeCmpPath.empty()) {
		auto soeCmp = loadDenseMatrix(soeCmpPath);
		for (Index ii = 0; ii < soeCmp.rows(); ii++) {
			cmps.emplace_back(soeCmp(ii, 0), soeCmp(ii, 1), soeCmp(ii, 2));
		}
	}

	// Construct the collection
	vector<size_t> subset;
	map<size_t, size_t> refIdx;
	for (Index i = 0; i < objects.size(); i++) {
		subset.push_back(objects(i));
		refIdx[objects(i)] = i;
	}
	log() << "Running FFT" << endl;
	vector<fft_idx> refFFT = fft(subset, refD);

	// Choose points for the final embedding
	// numToEmbed = min(nObj - subset.size(), numToEmbed);
	// numToEmbed = subset.size();
	numToEmbed = 0; // embed all points
	vector<size_t> embedded(subset);
	map<size_t, size_t> embIdx(refIdx);
	MatrixXd embX(refX);
	while (embedded.size() < nObj) {
		numToEmbed *= 2;
		numToEmbed = min(nObj - embedded.size(), numToEmbed);
		vector<size_t> embedPts;
		if (numToEmbed > 0 && numToEmbed < nObj - embedded.size()) {
			vector<size_t> options;
			for (size_t pt = 0; pt < nObj; pt++) {
				if (embIdx.find(pt) == embIdx.end()) {
					options.push_back(pt);
				}
			}
			auto ss = random_subset<size_t>(options.begin(), options.end(), numToEmbed);
			embedPts.insert(embedPts.begin(), ss.begin(), ss.end());
			embedPts.insert(embedPts.begin(), embedded.begin(), embedded.end());
			sort(embedPts.begin(), embedPts.end());
		} else {
			embedPts.resize(nObj);
			iota(embedPts.begin(), embedPts.end(), 0);
		}

		// Embed each point which is not in the reference
		log() << "Embedding " << embedPts.size() << " / " << nObj << " points" << endl;
		MatrixXd Xhat = MatrixXd::Random(embedPts.size(), nDim);
		vector<future<vector<pt_pos>>> futures;
		map<size_t, size_t> ptIdx;
		const size_t batch = std::thread::hardware_concurrency();
		auto from = embedPts.begin();
		auto to = from;
		for (size_t ib = 0; ib < batch; ib++) {
			from = to;
			if (ib == batch - 1) {
				to = embedPts.end();
			} else {
				to = next(from, embedPts.size() / batch);
			}
			vector<size_t> points;
			for (auto it = from; it != to; it++) {
				ptIdx[*it] = distance(embedPts.begin(), it);
				auto fit = embIdx.find(*it);
				if (fit == embIdx.end()) {
					points.push_back(*it);
				} else {
					// Save reference point positions
					Xhat.row(ptIdx[*it]) = embX.row(fit->second);
				}
			}
			futures.push_back(async(std::launch::async, embed, points, nDim,
				numAnchors, refX, subset, refFFT, oracle));
		}
		for (auto& future : futures) {
			for (const pt_pos& result : future.get()) {
				try {
					log() << "Saved " << result.pt << endl;
					Xhat.row(ptIdx[result.pt]) = result.pos;
					cmps.insert(cmps.end(), result.cmps.begin(), result.cmps.end());
					saveOrder(result);
				} catch(...) {
					// skip it, sad
				}
			}
		}

		if (numToEmbed > 0 && !soeCmpPath.empty()) {

			// Use SOE to finish the embedding
			log() << "Saving embedding" << endl;
			stringstream ss;
			ss << embPath << "_" << numToEmbed << ".pre.csv";
			saveEmbedding(Xhat, embedPts, ss.str());

			// Convert point indexes for the triples
			vector<CmpConstraint> localCmps(cmps);
			stringstream sscmp;
			sscmp << embPath << "_" << numToEmbed << ".cmps.csv";
			auto cmpFile = openSink(sscmp.str());
			for (auto& cmp : localCmps) {
				(*cmpFile) << cmp.a << "," << cmp.b << "," << cmp.c << endl;
				cmp.a = ptIdx[cmp.a];
				cmp.b = ptIdx[cmp.b];
				cmp.c = ptIdx[cmp.c];
				cmp.d = ptIdx[cmp.d];
			}
			cmpFile.reset();

			log() << "Using as " << Xhat.rows() << " x " << Xhat.cols()
				<< " initialization for SOE embedding" << endl;
			EmbedConfig config;
			config.nDim = nDim;
			config.minDelta = 1e-12;
			config.maxIter = 1e4;
			config.verbose = true;
			config.margin = 1e-2;
			auto result = embedCmpWithSOE(localCmps, Xhat, config);
			log() << "SOE embedding for " << numToEmbed << " achieved loss " << result.loss << endl;
			Xhat = result.X;

			auto Xrand = MatrixXd::Random(Xhat.rows(), Xhat.cols());
			auto result2 = embedCmpWithSOE(localCmps, Xrand, config);
			log() << "SOE for " << numToEmbed << " from random loss: " << result2.loss
				<< " init: " << result.loss << endl;
		}

		// Save the embedding.
		log() << "Saving embedding" << endl;
		// stringstream ss2;
		// ss2 << embPath << "_" << numToEmbed << ".post.csv";
		// saveEmbedding(Xhat, embedPts, ss2.str());
		saveEmbedding(Xhat, embedPts, embPath);

		// Prepare for the next round
		embedded = embedPts;
		embIdx = ptIdx;
		embX = Xhat;
	}

	log() << "Done" << endl;
	return 0;
}

/// Program entry point.
int main(int argc, char * argv[]) {
	try {
		stringstream distDesc;
		distDesc << "Distance function: " << DIST_EUCLIDEAN << ", " << DIST_COSINE << ", or " << DIST_JACCARD;
		options_description desc{"eval_embedding options"};
		desc.add_options()
			("help,h", "Show this usage message")
			("posfile", value<string>(), "Input file containing dataset")
			("ref", value<string>(), "Input embedding of reference subset")
			("num_pts", value<size_t>()->default_value(0), "Embed at most this many randomly-selected points")
			("dist", value<string>()->default_value("EUCLIDEAN"), distDesc.str().c_str())
			("soe_cmp", value<string>()->default_value(""), "Use SOE with the specified triples after the shell embedding")
			("embedding", value<string>(), "Output file for the embedding")
			("order", value<string>(), "Output file for the point embedding order and timing")
			("anchors", value<size_t>()->default_value(0), "Number of anchors to use")
			("noise", value<double>()->default_value(0), "Prob. of flipping a comparison outcome")
			;

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);

		bool valid = true;
		if (vm.count("help")) {
			cerr << desc << endl;
			valid = false;
		}
		if (!vm.count("posfile")) {
			cerr << "--posfile is required" << endl;
			valid = false;
		}
		if (!vm.count("ref")) {
			cerr << "--ref is required" << endl;
			valid = false;
		}
		if (!vm.count("embedding")) {
			cerr << "--embedding is required" << endl;
			valid = false;
		}
		if (!vm.count("order")) {
			cerr << "--order is required" << endl;
			valid = false;
		}
		if (valid) {
			return run_embed_from_ref(
				vm["posfile"].as<string>(),
				vm["ref"].as<string>(),
				vm["num_pts"].as<size_t>(),
				vm["dist"].as<string>(),
				vm["soe_cmp"].as<string>(),
				vm["embedding"].as<string>(),
				vm["order"].as<string>(),
				vm["anchors"].as<size_t>(),
				vm["noise"].as<double>());
		}
	} catch (const error &ex) {
		cerr << ex.what() << endl;
	}
	return 1;
}
