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

struct bfloat16
{
		uint16_t m_data;

		friend bool operator==(bfloat16 lhs, bfloat16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(bfloat16 lhs, bfloat16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};

struct float16
{
		uint16_t m_data;

		friend bool operator==(float16 lhs, float16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(float16 lhs, float16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};

namespace SIMD_NAMESPACE
{
#if SUPPORTS_AVX
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

	static inline __m256d combine(__m128d low, __m128d high = _mm_setzero_pd()) noexcept
	{
		return _mm256_setr_m128d(low, high);
	}
	static inline __m256 combine(__m128 low, __m128 high = _mm_setzero_ps()) noexcept
	{
		return _mm256_setr_m128(low, high);
	}
	static inline __m256i combine(__m128i low, __m128i high = _mm_setzero_si128()) noexcept
	{
		return _mm256_setr_m128i(low, high);
	}
#endif

#if SUPPORTS_AVX
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7>
	inline __m256i constant() noexcept
	{
		return _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m256i constant() noexcept
	{
		return constant<i0, i1, i0, i1, i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m256i constant() noexcept
	{
		return constant<i, i, i, i, i, i, i, i>();
	}
#elif SUPPORTS_SSE2
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
	inline __m128i constant() noexcept
	{
		return _mm_setr_epi32(i0, i1, i2, i3);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m128i constant() noexcept
	{
		return constant<i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m128i constant() noexcept
	{
		return constant<i, i, i, i>();
	}
#else
	template<uint32_t i>
	inline uint32_t constant() noexcept
	{
		return i;
	}
#endif

	template<typename T, typename U>
	inline constexpr T bitwise_cast(U x) noexcept
	{
		static_assert(sizeof(T) == sizeof(U));
		T result;
		std::memcpy(&result, &x, sizeof(T));
		return result;
	}
#if SUPPORTS_SSE41
	static inline int get_cutoff_mask(int num) noexcept
	{
		assert(num > 0);
		return (1 << num) - 1;
	}
#elif SUPPORTS_SSE2
	static inline __m128d get_cutoff_mask_pd(int num) noexcept
	{
		assert(num > 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_pd();
			case 1:
				return _mm_castsi128_pd(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFFu, 0x00000000, 0x00000000u));
			case 2:
			default:
				return _mm_castsi128_pd(_mm_set1_epi32(0xFFFFFFFFu));
		}
	}
	static inline __m128 get_cutoff_mask_ps(int num) noexcept
	{
		assert(num > 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_ps();
			case 1:
				return _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u));
			case 2:
				return _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u));
			case 3:
				return _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u));
			case 4:
			default:
				return _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFFu));
		}
	}
	static inline __m128i get_cutoff_mask_i(int num_bytes) noexcept
	{
		assert(num > 0);
		switch (num_bytes)
		{
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_setr_epi8(0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 2:
				return _mm_setr_epi16(0xFFFFu, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u);
			case 3:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 4:
				return _mm_setr_epi32(0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u);
			case 5:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 6:
				return _mm_setr_epi16(0xFFFFu, 0xFFFFu, 0xFFFFu, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u);
			case 7:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 8:
				return _mm_setr_epi32(0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u);
			case 9:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 10:
				return _mm_setr_epi16(0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0x0000u, 0x0000u, 0x0000u);
			case 11:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u);
			case 12:
				return _mm_setr_epi32(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u);
			case 13:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u, 0x00u, 0x00u);
			case 14:
				return _mm_setr_epi16(0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu, 0x0000u);
			case 15:
				return _mm_setr_epi8(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0x00u);
			case 16:
			default:
				return _mm_set1_epi32(0xFFFFFFFFu);
		}
	}
#endif

#if SUPPORTS_AVX
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
#elif SUPPORTS_SSE41
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
#elif SUPPORTS_SSE2
	static inline __m128 cutoff_ps(__m128 data, int num, __m128 value) noexcept
	{
		assert(num > 0);
		const __m128 mask = get_cutoff_mask_ps(num);
		return _mm_or_ps(_mm_and_ps(mask, data), _mm_andnot_ps(mask, value));
	}
	static inline __m128d cutoff_pd(__m128d data, int num, __m128d value) noexcept
	{
		assert(num > 0);
		const __m128d mask = get_cutoff_mask_pd(num);
		return _mm_or_pd(_mm_and_pd(mask, data), _mm_andnot_pd(mask, value));
	}
#else
	static inline float cutoff_ps(float data, int num, float value) noexcept
	{
		assert(num > 0);
		return (num == 0) ? value : data;
	}
	static inline double cutoff_pd(double data, int num, double value) noexcept
	{
		assert(num > 0);
		return (num == 0) ? value : data;
	}
#endif

}

#endif /* VECTORS_VECTOR_UTILS_HPP_ */
