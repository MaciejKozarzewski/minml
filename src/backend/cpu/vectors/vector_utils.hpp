/*
 * vector_utils.hpp
 *
 *  Created on: Jan 6, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_UTILS_HPP_
#define VECTORS_VECTOR_UTILS_HPP_

#include <cstring>
#include <cmath>
#include <cinttypes>
#include <x86intrin.h>
#include "vector_macros.hpp"

template<typename T, typename U>
T bitwise_cast(U x) noexcept
{
	static_assert(sizeof(T) == sizeof(U), "Cannot cast types of different sizes");
	union
	{
			U u;
			T t;
	} tmp;
	tmp.u = x;
	return tmp.t;
}

namespace SIMD_NAMESPACE
{
#if COMPILED_WITH_AVX
	static inline __m256i get_mask(int num) noexcept
	{
		assert(0 <= num && num <= 8);
// @formatter:off
		alignas(64) static const int32_t int32_masks[9][8] = {
		  {  0,  0,  0,  0,  0,  0,  0,  0 },
		  { -1,  0,  0,  0,  0,  0,  0,  0 },
		  { -1, -1,  0,  0,  0,  0,  0,  0 },
		  { -1, -1, -1,  0,  0,  0,  0,  0 },
		  { -1, -1, -1, -1,  0,  0,  0,  0 },
		  { -1, -1, -1, -1, -1,  0,  0,  0 },
		  { -1, -1, -1, -1, -1, -1,  0,  0 },
		  { -1, -1, -1, -1, -1, -1, -1,  0 },
		  { -1, -1, -1, -1, -1, -1, -1, -1 },
		};
// @formatter:on
		return _mm256_load_si256(reinterpret_cast<const __m256i*>(int32_masks[num]));
	}
#endif

}

#endif /* VECTORS_VECTOR_UTILS_HPP_ */
