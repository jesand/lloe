/// @file  random.hpp
/// @brief Declarations for random number generation

#pragma once
#ifndef OGT_UTIL_RANDOM_HPP
#define OGT_UTIL_RANDOM_HPP

#include <ogt/config.hpp>
#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace OGT_NAMESPACE {
namespace util {

/// The type used for the random engine
typedef std::mt19937 rand_gen;

/// Get the random engine
rand_gen& globalRand();

/// Seed the random engine with a fixed seed. Useful for testing.
void seedRand(int seed);

/// Flip a coin with a given bias
bool random_coin_flip(double bias);

/// Pick a random object from a container, and return an iterator to it.
template<class RandomAccessIterator>
auto random_choice(RandomAccessIterator begin, RandomAccessIterator end) {
	size_t numel = end - begin;
	std::uniform_int_distribution<> dist(0, numel - 1);
	size_t idx = dist(globalRand());
	return begin + idx;
}

/// Pick a random subset and return the zero-based subset indices.
std::set<size_t> random_nchoosek(size_t n, size_t k);

/// Pick a random subset of the rows of a matrix
template<typename T>
T random_rows(const T& matrix, size_t num, std::vector<size_t>* choice) {
	if (num >= static_cast<size_t>(matrix.rows())) {
		return matrix;
	}
	T submatrix(num, matrix.cols());
	auto list = random_nchoosek(matrix.rows(), num);
	std::shared_ptr<std::vector<size_t>> pchoice;
	if (!choice) {
		pchoice.reset(new std::vector<size_t>());
		choice = pchoice.get();
	}
	choice->clear();
	choice->insert(choice->end(), list.begin(), list.end());
	std::sort(choice->begin(), choice->end());
	size_t next = 0;
	for (size_t idx : *choice) {
		submatrix.row(next++) = matrix.row(idx);
	}
	return submatrix;
}

/// Pick a random subset of the rows and columns of a square matrix
template<typename T>
T random_rowscols(const T& matrix, size_t num, std::vector<size_t>* choice) {
	if (num >= static_cast<size_t>(matrix.rows())) {
		return matrix;
	}
	T submatrix(num, num);
	auto list = random_nchoosek(matrix.rows(), num);
	std::shared_ptr<std::vector<size_t>> pchoice;
	if (!choice) {
		pchoice.reset(new std::vector<size_t>());
		choice = pchoice.get();
	}
	choice->clear();
	choice->insert(choice->end(), list.begin(), list.end());
	std::sort(choice->begin(), choice->end());
	size_t row = 0;
	for (size_t ii : *choice) {
		size_t col = 0;
		for (size_t jj : *choice) {
			submatrix(row,col) = matrix(ii,jj);
			col++;
		}
		row++;
	}
	return submatrix;
}

/// Pick a random subset from a container
template<class T, class RandomAccessIterator>
std::set<T> random_subset(RandomAccessIterator begin, RandomAccessIterator end,
	size_t count) {
	std::set<T> res;
	for (size_t idx : random_nchoosek(end - begin, count)) {
		res.insert(*(begin + idx));
	}
	return res;
}

} // end namespace util
} // end namespace OGT_NAMESPACE
#endif /* OGT_UTIL_RANDOM_HPP */
