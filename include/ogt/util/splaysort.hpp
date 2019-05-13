/// @file  splaysort.hpp
/// @brief Adaptive sort algorithm, impemented by Moffat et al.

#pragma once
#ifndef OGT_UTIL_SPLAYSORT_HPP
#define OGT_UTIL_SPLAYSORT_HPP

#include <ogt/config.hpp>
#include <cstddef>
#include <functional>

extern "C" {
void
splaysort_impl(void *A, size_t n, size_t size,
	int (*cmp)(const void *, const void *));
}

namespace OGT_NAMESPACE {
namespace util {

template<class ItemType>
std::function<bool(ItemType a, ItemType b)> global_item_cmp;

template<class ItemType>
int global_cmp(const void* pa, const void* pb) {
	const ItemType a = *reinterpret_cast<const ItemType*>(pa);
	const ItemType b = *reinterpret_cast<const ItemType*>(pb);
	if (global_item_cmp<ItemType>(a, b)) {
		return -1;
	} else if (global_item_cmp<ItemType>(b, a)) {
		return +1;
	} else {
		return 0;
	}
}

template<class ItemType>
void splaysort( ItemType* first, size_t length,
	std::function<bool(ItemType a, ItemType b)> comp ) {
	global_item_cmp<ItemType> = comp;
	splaysort_impl(static_cast<void*>(first), length, sizeof(ItemType),
		global_cmp<ItemType>);
}


} // end namespace util
} // end namespace OGT_NAMESPACE
#endif /* OGT_UTIL_SPLAYSORT_HPP */
