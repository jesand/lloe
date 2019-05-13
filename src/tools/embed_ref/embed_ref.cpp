/// @file  embed_ref.cpp
/// @brief Choose and embed a reference subset.

#include <boost/program_options.hpp>
#include <ogt/ogt.hpp>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

using namespace boost::program_options;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using ogt::core::AB_LT_AC;
using ogt::core::Collection;
using ogt::core::createDistOracle;
using ogt::core::Oracle;
using ogt::embed::CmpConstraint;
using ogt::embed::embedCmpWithSOE;
using ogt::embed::EmbedConfig;
using ogt::embed::EmbedResult;
using ogt::io::IoErr;
using ogt::io::isSparseFile;
using ogt::io::loadDenseMatrix;
using ogt::io::loadSparseMatrix;
using ogt::io::openSink;
using ogt::io::saveEmbedding;
using ogt::linalg::DenseNZIterator;
using ogt::linalg::normalizePos;
using ogt::linalg::posToCosineDist;
using ogt::linalg::posToDist;
using ogt::linalg::posToJaccardDist;
using ogt::linalg::projectOntoUnitSphere;
using ogt::linalg::selectRows;
using ogt::util::random_nchoosek;
using std::async;
using std::cerr;
using std::cout;
using std::endl;
using std::future;
using std::make_shared;
using std::map;
using std::min;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

const char * DIST_EUCLIDEAN = "EUCLIDEAN";
const char * DIST_COSINE = "COSINE";
const char * DIST_JACCARD = "JACCARD";

std::ostream& log() {
	std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    cerr << std::put_time(&tm, "%c %Z") << ' ';
    return cerr;
}

/// Run a single embedding, perhaps in a thread pool.
shared_ptr<EmbedResult> embed(size_t nObj,
	const vector<CmpConstraint>& cmps, const EmbedConfig& config) {
	auto result = make_shared<EmbedResult>();
	result->X = MatrixXd::Random(nObj, config.nDim);
	*result = embedCmpWithSOE(cmps, result->X, config);
	return result;
}

/// Run the program.
int run_embed_ref(string posPath, string distType, bool projectOntoSphere,
	size_t subsetSize, size_t numRankings, size_t nDim, size_t batchSize, size_t bestOf,
	double target, bool doubleOnSuccess, double doublingTimeLimit,
	string embPath, string triplesPath) {

	// Load the true position matrix and create a distance oraclelog() "Loading position matrix from " << posPath << endl;
	MatrixXd truePos, subsetPos;
	Eigen::SparseMatrix<double, Eigen::RowMajor> spTruePos, spSubsetPos;
	size_t nObj;
	if (isSparseFile(posPath)) {
		try {
			spTruePos = loadSparseMatrix(posPath);
		} catch (const IoErr& e) {
			log() << "Could not load position matrix: " << e.what() << endl;
			return 0;
		}
		nObj = spTruePos.rows();
	} else {
		truePos = loadDenseMatrix(posPath);
		nObj = truePos.rows();
	}

	size_t numTries = 0;
	bool isDoubling = false;
	shared_ptr<EmbedResult> best;
	vector<size_t> subset;
	double savedLoss = INFINITY;
	auto startedSize = std::chrono::steady_clock::now();
	while (!best || (best->loss > target && numTries < bestOf)) {

		// Choose a subset at random
		log() << "Choosing subset from " << nObj << " total items" << endl;
		auto ss = random_nchoosek(nObj, subsetSize);
		subset.clear();
		subset.insert(subset.begin(), ss.begin(), ss.end());
		if (spTruePos.size() > 0) {
			spSubsetPos = selectRows(spTruePos, subset);
			log() << "Subset pos is " << spSubsetPos.rows() << " x " << spSubsetPos.cols() << endl;
		} else {
			subsetPos = selectRows(truePos, subset);
			log() << "Subset pos is " << subsetPos.rows() << " x " << subsetPos.cols() << endl;
		}

		// Construct the collection
		log() << "Finding all pairwise distances" << endl;
		MatrixXd dist;
		if (distType == DIST_EUCLIDEAN) {
			if (spSubsetPos.size() > 0) {
				dist = posToDist(spSubsetPos);
			} else {
				dist = posToDist(subsetPos);
			}
		} else if (distType == DIST_COSINE) {
			if (spSubsetPos.size() > 0) {
				dist = posToCosineDist(spSubsetPos);
			} else {
				dist = posToCosineDist(subsetPos);
			}
		} else if (distType == DIST_JACCARD) {
			if (spSubsetPos.size() > 0) {
				dist = posToJaccardDist<SparseMatrix<double, Eigen::RowMajor>, 
					Eigen::SparseVector<double>,
					Eigen::SparseVector<double>::InnerIterator>(
						spSubsetPos);
			} else {
				dist = posToJaccardDist<MatrixXd, VectorXd, DenseNZIterator<VectorXd>>(
						subsetPos);
			}
		} else {
			log() << "Invalid distance function: " << distType << endl;
			return 0;
		}
		log() << "Dist is " << dist.rows() << " x " << dist.cols() << endl;

		auto oracle = createDistOracle(dist);
		auto coll = Collection::Create(oracle, subset.size());

		// Choose comparisons for the subset.
		numRankings = std::min<size_t>(numRankings, subset.size());
		log() << "Finding comparisons for " << numRankings << " rankings" << endl;
		vector<CmpConstraint> cmps;
		for (size_t pt : coll->frftTraversal().stopAfter(numRankings)) {
			coll->sort(pt, cmps);
		}
		auto outTriples = openSink(triplesPath);
		if (!outTriples) {
			log() << "Could not open " << triplesPath << endl;
			return 1;
		}
		for (const auto& cmp : cmps) {
			*outTriples << subset[cmp.a] << ','
				<< subset[cmp.b] << ','
				<< subset[cmp.c] << endl;
		}
		outTriples.reset();
		log() << "Embedding from " << cmps.size() << " triples" << endl;

		// Embed the subset.
		log() << "Embedding subset into R^" << nDim << " from " << cmps.size() << " triples" << endl;
		EmbedConfig config;
		config.nDim = nDim;
		config.minDelta = 1e-12;
		config.maxIter = 1e4;
		config.verbose = false;
		config.margin = 1e-2;

		vector<future<shared_ptr<EmbedResult>>> futures;
		batchSize = min(batchSize, 1 + bestOf - numTries);
		if (!batchSize) {
			break;
		}
		auto started = std::chrono::steady_clock::now();
		for (size_t i = 0; i < batchSize; i++) {
			futures.push_back(async(embed, subsetSize, cmps, config));
		}
		for (auto& future : futures) {
			numTries++;
			auto result = future.get();
			if (!best || result->loss < best->loss) {
				best = result;
			}
			log() << "Got embedding loss " << result->loss << " in " << numTries << " tries; min is " << best->loss << endl;
			if (best->loss == 0) {
				break;
			} else if (isDoubling && doublingTimeLimit) {
				auto endedSize = std::chrono::steady_clock::now();
				std::chrono::duration<double> diff = endedSize - startedSize;
				if (diff.count() > doublingTimeLimit) {
					break;
				}
			}
		}
		auto ended = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = ended - started;
		log() << "Batch took " << diff.count() << " seconds" << endl;

		// Save the embedding.
		if (best->loss <= savedLoss) {
			savedLoss = best->loss;
			log() << "Saving embedding after " << numTries << " tries with loss "
				<< best->loss << endl;
			if (projectOntoSphere) {
				best->X = projectOntoUnitSphere(best->X);
			} else {
				best->X = normalizePos(best->X);
			}
			saveEmbedding(best->X, subset, embPath);
		}

		// Double on success
		if (isDoubling) {
			auto endedSize = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = endedSize - startedSize;
			log() << "Subset size took " << diff.count() << " seconds so far" << endl;
			if (isDoubling && doublingTimeLimit && diff.count() > doublingTimeLimit) {
				log() << "Stopping due to long size duration" << endl;
				break;
			}
		}
		if (doubleOnSuccess && best->loss <= savedLoss) {
			savedLoss = target;
			subsetSize = min(nObj, 2 * subsetSize);
			isDoubling = true;
			startedSize = std::chrono::steady_clock::now();
			log() << "Doubling subset size to " << subsetSize << endl;
			best.reset();
		}
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
			("dist", value<string>()->default_value("EUCLIDEAN"), distDesc.str().c_str())
			("project_onto_sphere", value<bool>()->default_value(false), "Project embedding onto centered sphere")
			("size", value<size_t>(), "Subset size")
			("num_rankings", value<size_t>(), "Number of rankings to embed with")
			("dim", value<size_t>(), "Embedding dimensionality")
			("embedding", value<string>(), "Output file for the reference embedding")
			("ref_triples", value<string>(), "Output file for the reference triples")
			("batch", value<size_t>()->default_value(1), "Re-roll after n tries")
			("best_of", value<size_t>()->default_value(1), "Return best of n runs")
			("ref_double", value<bool>()->default_value(false), "On success, attempt twice the subset size")
			("double_time_limit", value<double>()->default_value(0), "When attempting larger subsets, stop if a batch takes more than this many seconds")
			("target", value<double>()->default_value(0), "Stop when a loss at most this large is obtained")
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
		if (!vm.count("embedding")) {
			cerr << "--embedding is required" << endl;
			valid = false;
		}
		if (!vm.count("size")) {
			cerr << "--size is required" << endl;
			valid = false;
		}
		if (!vm.count("dim")) {
			cerr << "--dim is required" << endl;
			valid = false;
		}
		if (valid) {
			return run_embed_ref(
				vm["posfile"].as<string>(),
				vm["dist"].as<string>(),
				vm["project_onto_sphere"].as<bool>(),
				vm["size"].as<size_t>(),
				vm["num_rankings"].as<size_t>(),
				vm["dim"].as<size_t>(),
				vm["batch"].as<size_t>(),
				vm["best_of"].as<size_t>(),
				vm["target"].as<double>(),
				vm["ref_double"].as<bool>(),
				vm["double_time_limit"].as<double>(),
				vm["embedding"].as<string>(),
				vm["ref_triples"].as<string>());
		}
	} catch (const error &ex) {
		cerr << ex.what() << endl;
	}
	return 1;
}
