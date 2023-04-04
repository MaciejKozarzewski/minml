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
#if SUPPORTS_FP16
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
#if SUPPORTS_FP16
				return _cvtsh_ss(x.m_data);
#else
				return 0.0f;
#endif
			}
	};
	template<>
	struct Converter<SCALAR, float, sw_float16>
	{
			sw_float16 operator()(float x) const noexcept
			{
				return sw_float16 { 0 }; // TODO
			}
	};
	template<>
	struct Converter<SCALAR, sw_float16, float>
	{
			float operator()(sw_float16 x) const noexcept
			{
				return 0.0f; // TODO
			}
	};

	template<>
	struct Converter<SCALAR, float, bfloat16>
	{
			bfloat16 operator()(float x) const noexcept
			{
				return bfloat16 { 0 }; // TODO
			}
	};
	template<>
	struct Converter<SCALAR, bfloat16, float>
	{
			float operator()(bfloat16 x) const noexcept
			{
				return 0.0f; // TODO
			}
	};
	template<>
	struct Converter<SCALAR, float, sw_bfloat16>
	{
			sw_bfloat16 operator()(float x) const noexcept
			{
				const uint32_t bits = bitwise_cast<uint32_t>(x);
				return sw_bfloat16 { static_cast<uint16_t>(bits >> 16) };
			}
	};
	template<>
	struct Converter<SCALAR, sw_bfloat16, float>
	{
			float operator()(sw_bfloat16 x) const noexcept
			{
				const uint32_t bits = static_cast<uint32_t>(x.m_data) << 16;
				return bitwise_cast<float>(bits);
			}
	};

	/*
	 * SSE 128-bit code
	 */
#if SUPPORTS_SSE2
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
#  if SUPPORTS_F16C
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
#  if SUPPORTS_F16C
				return _mm_cvtph_ps(x);
#  else
				return _mm_setzero_ps();
#  endif
			}
	};
	template<>
	struct Converter<XMM, float, sw_float16>
	{
			__m128i operator()(__m128 x) const noexcept
			{
				return _mm_setzero_si128(); // TODO
			}
	};
	template<>
	struct Converter<XMM, sw_float16, float>
	{
			__m128 operator()(__m128i x) const noexcept
			{
				return _mm_setzero_ps(); // TODO
			}
	};

	template<>
	struct Converter<XMM, float, bfloat16>
	{
			__m128i operator()(__m128 x) const noexcept
			{
				return _mm_setzero_si128(); // TODO
			}
	};
	template<>
	struct Converter<XMM, bfloat16, float>
	{
			__m128 operator()(__m128i x) const noexcept
			{
				return _mm_setzero_ps(); // TODO
			}
	};
	template<>
	struct Converter<XMM, float, sw_bfloat16>
	{
			__m128i operator()(__m128 x) const noexcept
			{
#  if SUPPORTS_SSE41
				__m128i tmp = _mm_srli_epi32(_mm_castps_si128(x), 16); // shift right by 16 bits while shifting in zeros
				return _mm_packus_epi32(tmp, _mm_setzero_si128()); // pack 32 bits into 16 bits
#  else
				__m128i y0 = _mm_shufflelo_epi16(_mm_castps_si128(x), 0x0D);
				__m128i y1 = _mm_shufflehi_epi16(_mm_castps_si128(x), 0x0D);
				y0 = _mm_unpacklo_epi32(y0, _mm_setzero_si128());
				y1 = _mm_unpackhi_epi32(_mm_setzero_si128(), y1);
				return _mm_move_epi64(_mm_or_si128(y0, y1));
#  endif
			}
	};
	template<>
	struct Converter<XMM, sw_bfloat16, float>
	{
			__m128 operator()(__m128i x) const noexcept
			{
#  if SUPPORTS_SSE41
				__m128i tmp = _mm_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
				tmp = _mm_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#  else
				__m128i tmp = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
#  endif
				return _mm_castsi128_ps(tmp);
			}
	};
#endif

	/*
	 * AVX 256-bit code
	 */
#if SUPPORTS_AVX
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
#  if SUPPORTS_F16C
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
#  if SUPPORTS_F16C
				return _mm256_cvtph_ps(x);
#  else
				return _mm256_setzero_ps();
#  endif
			}
	};
	template<>
	struct Converter<YMM, float, sw_float16>
	{
			__m128i operator()(__m256 x) const noexcept
			{
				return _mm_setzero_si128(); // TODO
			}
	};
	template<>
	struct Converter<YMM, sw_float16, float>
	{
			__m256 operator()(__m128i x) const noexcept
			{
				return _mm256_setzero_ps(); // TODO
			}
	};

	template<>
	struct Converter<YMM, float, bfloat16>
	{
			__m128i operator()(__m256 x) const noexcept
			{
				return _mm_setzero_si128(); // TODO
			}
	};
	template<>
	struct Converter<YMM, bfloat16, float>
	{
			__m256 operator()(__m128i x) const noexcept
			{
				return _mm256_setzero_ps(); // TODO
			}
	};
	template<>
	struct Converter<YMM, float, sw_bfloat16>
	{
			__m128i operator()(__m256 x) const noexcept
			{
#  if SUPPORTS_AVX2
				__m256i tmp = _mm256_srli_epi32(_mm256_castps_si256(x), 16); // shift right by 16 bits while shifting in zeros
				return _mm_packus_epi32(get_low(tmp), get_high(tmp)); // pack 32 bits into 16 bits
#  else
				__m128i tmp_lo = _mm_srli_epi32(_mm_castps_si128(get_low(x)), 16); // shift right by 16 bits while shifting in zeros
				__m128i tmp_hi = _mm_srli_epi32(_mm_castps_si128(get_high(x)), 16); // shift right by 16 bits while shifting in zeros
				return _mm_packus_epi32(tmp_lo, tmp_hi); // pack 32 bits into 16 bits
#  endif
			}
	};
	template<>
	struct Converter<YMM, sw_bfloat16, float>
	{
			__m256 operator()(__m128i x) const noexcept
			{
#  if SUPPORTS_AVX2
				__m256i tmp = _mm256_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
				tmp = _mm256_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#  else
				__m128i tmp_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
				__m128i tmp_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), x); // pad upper half with zeros
				__m256i tmp = _mm256_setr_m128i(tmp_lo, tmp_hi); // combine two halves
#  endif
				return _mm256_castsi256_ps(tmp);
			}
	};
#endif

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_TYPE_CONVERSIONS_HPP_ */
