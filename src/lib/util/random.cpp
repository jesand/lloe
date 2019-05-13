/// @file  random.cpp
/// @brief Random number generation definitions.

#include <ogt/util/random.hpp>

namespace OGT_NAMESPACE {
namespace util {

/// The unique random engine
static rand_gen rd;

/// Whether the random engine needs to be seeded
static bool rdSeeded = false;

/// Get the random engine
rand_gen& globalRand() {
	if (!rdSeeded) {
		std::random_device rdev;
		rd.seed(rdev());
		rdSeeded = true;
	}
	return rd;
}

/// Seed the random engine with a fixed seed. Useful for testing.
void seedRand(int seed) {
	std::seed_seq sq{seed};
	rd.seed(sq);
	rdSeeded = true;
}

/// Flip a coin with a given bias
bool random_coin_flip(double bias) {
	return std::bernoulli_distribution(bias)(globalRand());
}

/// Pick a random subset and return the zero-based subset indices.
std::set<size_t> random_nchoosek(size_t n, size_t k) {
	std::set<size_t> choice;
	if (k >= n) {
		for (size_t i = 0; i < n; i++) {
			choice.insert(i);
		}
	} else {
		std::vector<size_t> order(n);
		std::iota(order.begin(), order.end(), 0);
		auto& rng = globalRand();
		std::shuffle(order.begin(), order.end(), rng);
		choice.insert(order.begin(), order.begin() + k);
	}
	return choice;
}

} // end namespace util
} // end namespace OGT_NAMESPACE
