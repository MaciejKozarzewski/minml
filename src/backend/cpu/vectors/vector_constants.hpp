/*
 * vector_constants.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_CONSTANTS_HPP_
#define VECTORS_VECTOR_CONSTANTS_HPP_

#include "vector_utils.hpp"
#include "register_type.hpp"

namespace SIMD_NAMESPACE
{
	template<uint32_t i>
	inline uint32_t scalar_constant() noexcept
	{
		return i;
	}

#if SUPPORTS_SSE2
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
	inline __m128i xmm_constant() noexcept
	{
		return _mm_setr_epi32(i0, i1, i2, i3);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m128i xmm_constant() noexcept
	{
		return xmm_constant<i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m128i xmm_constant() noexcept
	{
		return _mm_set1_epi32(i);
	}
#endif

#if SUPPORTS_AVX
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7>
	inline __m256i ymm_constant() noexcept
	{
		return _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m256i ymm_constant() noexcept
	{
		return ymm_constant<i0, i1, i0, i1, i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m256i ymm_constant() noexcept
	{
		return _mm256_set1_epi32(i);
	}
#endif

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_VECTOR_CONSTANTS_HPP_ */
