/*
 * type_conversions.hpp
 *
 *  Created on: Mar 29, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_TYPE_CONVERSIONS_HPP_
#define VECTORS_TYPE_CONVERSIONS_HPP_

#include "vector_macros.hpp"
#include "vector_utils.hpp"
#include "types.hpp"
#include "register_type.hpp"

#include <x86intrin.h>

namespace SIMD_NAMESPACE
{

	template<RegisterType RT, typename From, typename To>
	struct Converter;

	/*
	 * Scalar code
	 */
	template<typename From, typename To>
	struct Converter<SCALAR, From, To>
	{
			To operator()(From x) const noexcept
			{
				return static_cast<To>(x);
			}
	};
	template<>
	struct Converter<SCALAR, float, float16>
	{
			float16 operator()(float x) const noexcept
			{
#if COMPILED_WITH_F16C
				return float16 { _cvtss_sh(x, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)) };
#else
				return float16 { 0u };
#endif
			}
	};
	template<>
	struct Converter<SCALAR, float16, float>
	{
			float operator()(float16 x) const noexcept
			{
#if COMPILED_WITH_F16C
				return _cvtsh_ss(x.m_data);
#else
				return 0.0f;
#endif
			}
	};

	/*
	 * SSE 128-bit code
	 */
#if COMPILED_WITH_SSE2
	template<typename T>
	struct Converter<XMM, T, T>
	{
			template<typename U>
			U operator()(U x) const noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<XMM, float, float16>
	{
			__m128i operator()(__m128 x) const noexcept
			{
#  if COMPILED_WITH_F16C
				return _mm_cvtps_ph(x, _MM_FROUND_NO_EXC);
#  else
				return _mm_setzero_si128();
#  endif
			}
	};
	template<>
	struct Converter<XMM, float16, float>
	{
			__m128 operator()(__m128i x) const noexcept
			{
#  if COMPILED_WITH_F16C
				return _mm_cvtph_ps(x);
#  else
				return _mm_setzero_ps();
#  endif
			}
	};
#endif

/*
 * AVX 256-bit code
 */
#if COMPILED_WITH_AVX
	template<typename T>
	struct Converter<YMM, T, T>
	{
			template<typename U>
			U operator()(U x) const noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<YMM, float, float16>
	{
			__m128i operator()(__m256 x) const noexcept
			{
#  if COMPILED_WITH_F16C
				return _mm256_cvtps_ph(x, _MM_FROUND_NO_EXC);
#  else
				return _mm_setzero_si128();
#  endif
			}
	};
	template<>
	struct Converter<YMM, float16, float>
	{
			__m256 operator()(__m128i x) const noexcept
			{
#  if COMPILED_WITH_F16C
				return _mm256_cvtph_ps(x);
#  else
				return _mm256_setzero_ps();
#  endif
			}
	};
#endif

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_TYPE_CONVERSIONS_HPP_ */
