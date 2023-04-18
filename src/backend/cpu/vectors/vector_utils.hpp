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
#include "vector_masks.hpp"

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
	static inline __m128d get_low(__m256d reg) noexcept
	{
		return _mm256_castpd256_pd128(reg);
	}
	static inline __m128 get_low(__m256 reg) noexcept
	{
		return _mm256_castps256_ps128(reg);
	}
	static inline __m128i get_low(__m256i reg) noexcept
	{
		return _mm256_castsi256_si128(reg);
	}

	static inline __m128d get_high(__m256d reg) noexcept
	{
		return _mm256_extractf128_pd(reg, 1);
	}
	static inline __m128 get_high(__m256 reg) noexcept
	{
		return _mm256_extractf128_ps(reg, 1);
	}
	static inline __m128i get_high(__m256i reg) noexcept
	{
		return _mm256_extractf128_si256(reg, 1);
	}

	static inline __m256d combine(__m128d low, __m128d high) noexcept
	{
		return _mm256_setr_m128d(low, high);
	}
	static inline __m256 combine(__m128 low, __m128 high) noexcept
	{
		return _mm256_setr_m128(low, high);
	}
	static inline __m256i combine(__m128i low, __m128i high) noexcept
	{
		return _mm256_setr_m128i(low, high);
	}
#endif

#if COMPILED_WITH_AVX
	static inline __m256d cutoff_pd(__m256d data, int num, __m256d value) noexcept
	{
		assert(num > 0);
		switch (num)
		{
			case 0:
				return value;
			case 1:
				return _mm256_blend_pd(value, data, 1);
			case 2:
				return _mm256_blend_pd(value, data, 3);
			case 3:
				return _mm256_blend_pd(value, data, 7);
			default:
			case 4:
				return data;
		}
	}
	static inline __m256 cutoff_ps(__m256 data, int num, __m256 value) noexcept
	{
		assert(num > 0);
		switch (num)
		{
			case 0:
				return value;
			case 1:
				return _mm256_blend_ps(value, data, 1);
			case 2:
				return _mm256_blend_ps(value, data, 3);
			case 3:
				return _mm256_blend_ps(value, data, 7);
			case 4:
				return _mm256_blend_ps(value, data, 15);
			case 5:
				return _mm256_blend_ps(value, data, 31);
			case 6:
				return _mm256_blend_ps(value, data, 63);
			case 7:
				return _mm256_blend_ps(value, data, 127);
			default:
			case 8:
				return data;
		}
	}
#elif COMPILED_WITH_SSE41
	static inline __m128 cutoff_ps(__m128 data, int num, __m128 value) noexcept
	{
		assert(num > 0);
		switch(num)
		{
			case 0:
				return value;
			case 1:
				return _mm_blend_ps(value, data, 1);
			case 2:
				return _mm_blend_ps(value, data, 3);
			case 3:
				return _mm_blend_ps(value, data, 7);
			default:
			case 4:
				return data;
		}
	}
	static inline __m128d cutoff_pd(__m128d data, int num, __m128d value) noexcept
	{
		switch(num)
		{
			case 0:
				return value;
			case 1:
				return _mm_blend_pd(value, data, 1);
			default:
			case 2:
				return data;
		}
	}
#elif COMPILED_WITH_SSE
	static inline __m128 cutoff_ps(__m128 data, int num, __m128 value) noexcept
	{
		const __m128 mask = get_cutoff_mask_ps(num);
		return _mm_or_ps(_mm_and_ps(mask, data), _mm_andnot_ps(mask, value));
	}
	static inline __m128d cutoff_pd(__m128d data, int num, __m128d value) noexcept
	{
		const __m128d mask = get_cutoff_mask_pd(num);
		return _mm_or_pd(_mm_and_pd(mask, data), _mm_andnot_pd(mask, value));
	}
#else
	static inline float cutoff_ps(float data, int num, float value) noexcept
	{
		assert(num == 0 || num == 1);
		return (num == 0) ? value : data;
	}
	static inline double cutoff_pd(double data, int num, double value) noexcept
	{
		assert(num == 0 || num == 1);
		return (num == 0) ? value : data;
	}
#endif

}

#endif /* VECTORS_VECTOR_UTILS_HPP_ */
