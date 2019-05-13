/// @file  embed_cmp.cpp
/// @brief Embeds an object collection using oracle comparisons.

#include <boost/program_options.hpp>
#include <ogt/ogt.hpp>
#include <algorithm>
#include <chrono>
#include <map>

using Eigen::Index;
using Eigen::MatrixXd;
using namespace boost::program_options;
using nlohmann::json;
using OGT_NAMESPACE::core::CmpOutcome;
using OGT_NAMESPACE::embed::CmpConstraint;
using OGT_NAMESPACE::embed::EmbedConfig;
using OGT_NAMESPACE::embed::EmbedResult;
using OGT_NAMESPACE::io::loadCsvMatrix;
using OGT_NAMESPACE::io::openSink;
using OGT_NAMESPACE::io::openSource;
using OGT_NAMESPACE::io::saveCsvMatrix;
using OGT_NAMESPACE::io::saveEmbedding;
using OGT_NAMESPACE::linalg::embeddingFromKernelSVD;
using std::cerr;
using std::cout;
using std::endl;
using std::map;
using std::max;
using std::string;
using std::stringstream;
using std::vector;

/// Embed to a kernel using CK
const string METHOD_CK_K("ck_k");

/// Embed using CK
const string METHOD_CK_X("ck_x");

/// Embed to a kernel using GNMDS
const string METHOD_GNMDS_K("gnmds_k");

/// Embed using GNMDS
const string METHOD_GNMDS_X("gnmds_x");

/// Embed using Soft Ordinal Embedding
const string METHOD_SOE("soe");

/// Embed to a kernel using STE
const string METHOD_STE_K("ste_k");

/// Embed using STE
const string METHOD_STE_X("ste_x");

/// Embed using t-STE
const string METHOD_TSTE("tste");

/// Embed using PGD-X
const string METHOD_PGD_X("pgd_x");

/// Embed using PGD-K
const string METHOD_PGD_K("pgd_k");

/// Embed using NNPGD
const string METHOD_NNPGD("nnpgd");

/// Embed using NNPGD-debiased
const string METHOD_NNPGD_DEBIASED("nnpgd_debiased");

/// Embed using tau-for-k method
const string METHOD_TAU_K("tau_k");

/// Main program.
int run_embed_cmp(
	string cmp,
	size_t nDim,
	string method,
	string initPos,
	double margin,
	double lambda,
	double alpha,
	double minDelta,
	size_t maxIter,
	size_t bestOf,
	bool saveAll,
	double bestTol,
	bool verbose,
	string out,
	string outk,
	string outstats) {
	json results;
	results["args"] = {
		{"cmp", cmp},
		{"nDim", nDim},
		{"method", method},
		{"initPos", initPos},
		{"margin", margin},
		{"lambda", lambda},
		{"alpha", alpha},
		{"minDelta", minDelta},
		{"maxIter", maxIter},
		{"bestOf", bestOf},
		{"saveAll", saveAll},
		{"bestTol", bestTol},
		{"verbose", verbose},
		{"out", out},
		{"outk", outk},
		{"outstats", outstats},
	};
	EmbedConfig config;
	config.nDim = nDim;
	config.minDelta = minDelta;
	config.maxIter = maxIter;
	config.verbose = verbose;
	config.margin = margin;

	// Load the triples
	vector<CmpConstraint> cons;
	auto incmp = openSource(cmp);
	if (!incmp || !*incmp) {
		cerr << "Could not load constraints from: " << cmp << endl;
		return 1;
	}
	CmpOutcome con;
	map<size_t,size_t> points; // Map from true ID to temp ID
	vector<size_t> pointIds; // Map from temp ID to true ID
	size_t maxObj = 0;
	while (*incmp >> con) {
		for (size_t pt : {con.a, con.b, con.c}) {
			auto it = points.find(pt);
			if (it == points.end()) {
				points[pt] = pointIds.size();
				pointIds.push_back(pt);
				maxObj = max(maxObj, pt);
			}
		}
		cons.emplace_back(con);
	}
	size_t nObj = pointIds.size();
	cout << "Embedding " << nObj << " points referenced by " << cons.size() << " triples" << endl;
	results["args"]["nObj"] = nObj;
	results["num_triples"] = cons.size();

	// Rename objects as needed to embed only what we have constrained
	if (nObj < maxObj + 1) {
		for (auto& con : cons) {
			con.a = points[con.a];
			con.b = points[con.b];
			con.c = points[con.c];
			con.d = points[con.d];
		}
	} else {
		pointIds.clear();
	}

	// Load the initial positions, if we have an init file
	MatrixXd X0(nObj, nDim);
	if (!initPos.empty()) {
		try {
			auto X = loadCsvMatrix(initPos);
			if (pointIds.empty()) {
				X0 = X;
			} else {
				for (Index ii = 0; ii < X0.rows(); ii++) {
					X0.row(ii) = X.row(points[ii]);
				}
			}
		} catch (std::exception &e) {
			cerr << "Could not load initial positions: " << e.what() << endl;
			return 1;
		}
	}
	points.clear();

	auto save = [&](EmbedResult& res, string pathX, string pathK) {

		// Save the embedding
		bool gotKernel = (res.K.rows() > 0);
		try {
			if (gotKernel) {
				cout << "Calculating embedding from kernel..." << endl;
				res.X = embeddingFromKernelSVD(res.K, config.nDim);
			}
			if (!pointIds.empty()) {
				cout << "Writing " << pathX << endl;
				saveEmbedding(res.X, pointIds, pathX);
			} else {
				cout << "Writing " << pathX << endl;
				saveEmbedding(res.X, pathX);
			}
		} catch (const std::exception& e) {
			cerr << e.what() << endl;
			return;
		}

		// Save the kernel
		if (!pathK.empty()) {
			try {
				cout << "Writing " << pathK << endl;
				if (res.K.rows() == Eigen::Dynamic) {
					res.K = res.X * res.X.transpose();
				}
				saveCsvMatrix(res.K, pathK);
			} catch (const std::exception& e) {
				cerr << e.what() << endl;
				return;
			}
		}
	};

	// Produce the embedding
	EmbedResult bestRes;
	bestOf = max<size_t>(bestOf, 1);
	bestRes.loss = 1000 + bestTol;
	for (size_t rep = 0; rep < bestOf && (saveAll || bestRes.loss > bestTol); rep++) {
		if (initPos.empty()) {
			X0 = MatrixXd::Random(nObj, nDim);
		}

		cout << "Rep " << rep+1 << " / " << bestOf << ": Embedding " << nObj
			<< " objects into R^" << config.nDim << " using "
			<< cons.size() << " constraints" << endl;
		EmbedResult res;
		auto startTime = std::chrono::steady_clock::now();
		auto startProc = clock();
		if (method == METHOD_CK_K) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithCKForK(cons, K0, lambda, config);
		} else if (method == METHOD_CK_X) {
			res = embedCmpWithCKForX(cons, X0, lambda, config);
		} else if (method == METHOD_GNMDS_K) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithGNMDSForK(cons, K0, lambda, config);
		} else if (method == METHOD_GNMDS_X) {
			res = embedCmpWithGNMDSForX(cons, X0, lambda, config);
		} else if (method == METHOD_SOE) {
			res = embedCmpWithSOE(cons, X0, config);
		} else if (method == METHOD_STE_K) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithSTEForK(cons, K0, lambda, config);
		} else if (method == METHOD_STE_X) {
			res = embedCmpWithSTEForX(cons, X0, lambda, config);
		} else if (method == METHOD_TSTE) {
			res = embedCmpWithTSTE(cons, X0, lambda, alpha, config);
		} else if (method == METHOD_PGD_X) {
			res = embedCmpWithPGDForX(cons, X0, config);
		} else if (method == METHOD_PGD_K) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithPGDForK(cons, K0, config);
		} else if (method == METHOD_NNPGD) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithNNPGD(cons, K0, lambda, config);
		} else if (method == METHOD_NNPGD_DEBIASED) {
			MatrixXd K0 = X0 * X0.transpose();
			res = embedCmpWithNNPGDDebiased(cons, K0, lambda, config);
		} else if (method == METHOD_TAU_K) {
			res = embedCmpWithTauForK(cons, lambda, 1 - lambda);
		} else {
			cerr << "Invalid embedding method \"" << method << "\"" << endl;
			return 1;
		}
		auto endTime = std::chrono::steady_clock::now();
		auto endProc = clock();
		std::chrono::duration<double> diffTime = (endTime - startTime);
		double durTime = diffTime.count();
		double durProc = (endProc - startProc) / CLOCKS_PER_SEC;

		cout << "Rep " << rep+1 << " / " << bestOf << ": Duration: " << durProc
			<< " CPU, " << durTime << " wall" << endl;
		cout << "Rep " << rep+1 << " / " << bestOf << ": Objective: "
			<< res.loss << endl;
		if (rep == 0 || bestRes.loss > res.loss) {
			bestRes = res;
			results["elapsed_perf_counter"] = durProc;
			results["elapsed_process_time"] = durTime;
			results["loss"] = res.loss;
		}

		if (saveAll) {
			stringstream outX, outK;
			outX << out << ".try_" << rep+1 << ".csv.gz";
			outK << outk << ".try_" << rep+1 << ".csv.gz";
			save(res, outX.str(), outK.str());
		}
	}
	save(bestRes, out, outk);

	// Save our statistics
	if (!outstats.empty()) {
		cout << "Writing " << outstats << endl;
		auto fstats = openSink(outstats);
		if (!fstats || !fstats->good()) {
			cerr << "Could not open statistics output file" << endl;
			return 1;
		}
		*fstats << std::setw(2) << results << endl;
	}

	cout << "Done" << endl;
	return 0;
}

/// Program entry point.
int main(int argc, char * argv[]) {
	try {
		string methodDesc = string("Embedding method: ")
			+ METHOD_CK_K + ", " + METHOD_CK_X + ", "
			+ METHOD_GNMDS_X + ", " + METHOD_GNMDS_K
		 	+ ", " + METHOD_SOE
		 	+ ", " + METHOD_STE_K + ", " + METHOD_STE_X + ", " + METHOD_TSTE
		 	+ ", " + METHOD_PGD_X + ", " + METHOD_PGD_K + ", " + METHOD_NNPGD
		 	+ ", " + METHOD_NNPGD_DEBIASED
		 	+ ", or " + METHOD_TAU_K;
		options_description desc{"embed_cmp options"};
		desc.add_options()
			("help,h", "Show this usage message")
			("cmp", value<string>(), "Path to a file containing comparisons to embed (*.csv)")
			("ndim", value<size_t>(), "Embedding dimensionality")
			("method", value<string>(), methodDesc.c_str())
			("init_pos", value<string>()->default_value(""), "Initialize from the specified positions instead of at random (*.csv)")
			("margin", value<double>()->default_value(0), "Constraint margin")
			("lambda", value<double>()->default_value(0), "Mixing parameter for loss functions")
			("alpha", value<double>()->default_value(0), "Degrees of freedom for loss functions")
			("min_delta", value<double>()->default_value(1e-12), "Min loss objective delta to stop")
			("max_iter", value<size_t>()->default_value(1000), "Max iteration count")
			("best_of", value<size_t>()->default_value(1), "Return best of n runs")
			("best_tol", value<double>()->default_value(0), "Stop when a run achieves this score")
			("save_all", value<bool>()->default_value(false), "Save intermediate embeddings")
			("verbose", value<bool>()->default_value(false), "Verbose output")
			("out", value<string>(), "Output path for the embedding (*.csv)")
			("outk", value<string>()->default_value(""), "Output path for the kernel (*.csv)")
			("outstats", value<string>()->default_value(""), "Output path for statistics (*.json)")
			;

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);

		bool valid = true;
		if (vm.count("help")) {
			cout << desc << endl;
			valid = false;
		}
		if (!vm.count("cmp")) {
			cerr << "--cmp is required" << endl;
			valid = false;
		}
		if (!vm.count("ndim")) {
			cerr << "--ndim is required" << endl;
			valid = false;
		}
		if (!vm.count("method")) {
			cerr << "--method is required" << endl;
			valid = false;
		}
		if (!vm.count("out")) {
			cerr << "--out is required" << endl;
			valid = false;
		}
		if (valid) {
			return run_embed_cmp(
				vm["cmp"].as<string>(),
				vm["ndim"].as<size_t>(),
				vm["method"].as<string>(),
				vm["init_pos"].as<string>(),
				vm["margin"].as<double>(),
				vm["lambda"].as<double>(),
				vm["alpha"].as<double>(),
				vm["min_delta"].as<double>(),
				vm["max_iter"].as<size_t>(),
				vm["best_of"].as<size_t>(),
				vm["save_all"].as<bool>(),
				vm["best_tol"].as<double>(),
				vm["verbose"].as<bool>(),
				vm["out"].as<string>(),
				vm["outk"].as<string>(),
				vm["outstats"].as<string>());
		}
	} catch (const error &ex) {
		cerr << ex.what() << endl;
	}
	return 1;
}
