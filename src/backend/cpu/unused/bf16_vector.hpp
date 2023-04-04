/*
 * bf16_vector.hpp
 *
 *  Created on: Nov 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_BF16_VECTOR_HPP_
#define VECTORS_BF16_VECTOR_HPP_

#include <cstring>

//#include "fp32_vector.hpp"
//#include "generic_vector.hpp"
//#include "vector_length.hpp"
//#include "vector_load_store.hpp"
//#include "vector_utils.hpp"
//
//namespace scalar
//{
//	static inline float bfloat16_to_float(bfloat16 x) noexcept
//	{
//		const uint32_t bits = static_cast<uint32_t>(x.m_data) << 16;
//		return bitwise_cast<float>(bits);
//	}
//	static inline bfloat16 float_to_bfloat16(float x) noexcept
//	{
//		const uint32_t bits = bitwise_cast<uint32_t>(x);
//		return bfloat16 { static_cast<uint16_t>(bits >> 16) };
//	}
//}
//
//namespace SIMD_NAMESPACE
//{
//#if SUPPORTS_AVX
//
//	static inline __m256 bfloat16_to_float(__m128i x) noexcept
//	{
//#  if SUPPORTS_AVX2
//		__m256i tmp = _mm256_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
//		tmp = _mm256_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
//#  else
//		__m128i tmp_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
//		__m128i tmp_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), x); // pad upper half with zeros
//		__m256i tmp = _mm256_setr_m128i(tmp_lo, tmp_hi); // combine two halves
//#  endif
//		return _mm256_castsi256_ps(tmp);
//	}
//	static inline __m128i float_to_bfloat16(__m256 x) noexcept
//	{
//#  if SUPPORTS_AVX2
//		__m256i tmp = _mm256_srli_epi32(_mm256_castps_si256(x), 16); // shift right by 16 bits while shifting in zeros
//		return _mm_packus_epi32(get_low(tmp), get_high(tmp)); // pack 32 bits into 16 bits
//#  else
//		__m128i tmp_lo = _mm_srli_epi32(_mm_castps_si128(get_low(x)), 16); // shift right by 16 bits while shifting in zeros
//		__m128i tmp_hi = _mm_srli_epi32(_mm_castps_si128(get_high(x)), 16); // shift right by 16 bits while shifting in zeros
//		return _mm_packus_epi32(tmp_lo, tmp_hi); // pack 32 bits into 16 bits
//#  endif
//	}
//
//#elif SUPPORTS_SSE2
//
//	static inline __m128 bfloat16_to_float(__m128i x) noexcept
//	{
//#  if SUPPORTS_SSE41
//		__m128i tmp = _mm_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
//		tmp = _mm_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
//#  else
//		__m128i tmp = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
//#  endif
//		return _mm_castsi128_ps(tmp);
//	}
//	static inline __m128i float_to_bfloat16(__m128 x) noexcept
//	{
//#  if SUPPORTS_SSE41
//		__m128i tmp = _mm_srli_epi32(_mm_castps_si128(x), 16); // shift right by 16 bits while shifting in zeros
//		return _mm_packus_epi32(tmp, _mm_setzero_si128()); // pack 32 bits into 16 bits
//#else
//		__m128i y0 = _mm_shufflelo_epi16(_mm_castps_si128(x), 0x0D);
//		__m128i y1 = _mm_shufflehi_epi16(_mm_castps_si128(x), 0x0D);
//		y0 = _mm_unpacklo_epi32(y0, _mm_setzero_si128());
//		y1 = _mm_unpackhi_epi32(_mm_setzero_si128(), y1);
//		return _mm_move_epi64(_mm_or_si128(y0, y1));
//#endif
//	}
//
//#else
//
//	static inline float bfloat16_to_float(bfloat16 x) noexcept
//	{
//		return scalar::bfloat16_to_float(x);
//	}
//	static inline bfloat16 float_to_bfloat16(float x) noexcept
//	{
//		return scalar::float_to_bfloat16(x);
//	}
//
//#endif
//
//	template<>
//	class Vector<bfloat16>
//	{
//		private:
//#if SUPPORTS_AVX
//			__m256 m_data;
//#elif SUPPORTS_SSE2
//			__m128 m_data;
//#else
//			float m_data;
//#endif
//		public:
//			static constexpr int length = vector_length<bfloat16>();
//
//			Vector() noexcept // @suppress("Class members should be properly initialized")
//			{
//			}
//			Vector(const float *ptr, int num = length) noexcept :
//					m_data(vector_load(ptr, num))
//			{
//			}
//			Vector(const bfloat16 *ptr, int num = length) noexcept
//			{
//			}
//			Vector(Vector<float> x) noexcept :
//					m_data(x)
//			{
//			}
//			Vector(float x) noexcept
//			{
//#if SUPPORTS_AVX
//				m_data = _mm256_set1_ps(x);
//#elif SUPPORTS_SSE2
//				m_data = _mm_set1_ps(x);
//#else
//				m_data = x;
//#endif
//			}
//			Vector(bfloat16 x) noexcept
//			{
//#if SUPPORTS_SSE2
//				m_data = bfloat16_to_float(_mm_set1_epi16(x.m_data));
//#else
//				m_data = scalar::bfloat16_to_float(x);
//#endif
//			}
//			Vector(double x) noexcept :
//					Vector(static_cast<float>(x))
//			{
//			}
//			operator Vector<float>() const noexcept
//			{
//				return Vector<float>(m_data); // @suppress("Ambiguous problem")
//			}
//			void load(const float *ptr, int num) noexcept
//			{
//				m_data = vector_load(ptr, num);
//			}
//			void load(const bfloat16 *ptr, int num = length) noexcept
//			{
//			}
//			void store(float *ptr, int num = length) const noexcept
//			{
//				vector_store(m_data, ptr, num);
//			}
//			void store(bfloat16 *ptr, int num = length) const noexcept
//			{
//			}
//			void insert(float value, int index) noexcept
//			{
////				m_data.insert(value, index); // FIXME
//			}
//			float extract(int index) const noexcept
//			{
//#if SUPPORTS_AVX512
//				return 0.0f; // FIXME
//#else
//				assert(index >= 0 && index < length);
//				float tmp[length];
//				store(tmp);
//				return tmp[index];
//#endif
//			}
//			float operator[](int index) const noexcept
//			{
//				return extract(index);
//			}
//			void cutoff(const int num, Vector<bfloat16> value = zero()) noexcept
//			{
//				m_data = cutoff_ps(m_data, num, value.m_data);
//			}
//
//			static constexpr bfloat16 scalar_zero() noexcept
//			{
//				return bfloat16 { 0x0000u };
//			}
//			static constexpr bfloat16 scalar_one() noexcept
//			{
//				return bfloat16 { 0x3f80u };
//			}
//			static constexpr bfloat16 scalar_epsilon() noexcept
//			{
//				return bfloat16 { 0x0800u }; // FIXME this is not correct
//			}
//
//			static Vector<bfloat16> zero() noexcept
//			{
//				return Vector<bfloat16>(scalar_zero());
//			}
//			static Vector<bfloat16> one() noexcept
//			{
//				return Vector<bfloat16>(scalar_one());
//			}
//			static Vector<bfloat16> epsilon() noexcept
//			{
//				return Vector<bfloat16>(scalar_epsilon());
//			}
//	};
//
//	/*
//	 * Float vector logical operations.
//	 * Return vector of half floats, either 0x0000 (0.0f) for false, or 0xFFFF (-nan) for true.
//	 */
//	static inline Vector<bfloat16> operator==(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) == static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator!=(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) != static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator<(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) < static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator<=(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) <= static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator>(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) > static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator>=(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) >= static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator&(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) & static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator|(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) | static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator^(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) ^ static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator~(Vector<bfloat16> x) noexcept
//	{
//		return ~static_cast<Vector<float>>(x);
//	}
//	static inline Vector<bfloat16> operator!(Vector<bfloat16> x) noexcept
//	{
//		return ~x;
//	}
//
//	/*
//	 * Float vector arithmetics.
//	 */
//	static inline Vector<bfloat16> operator+(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) + static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator-(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) - static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator-(Vector<bfloat16> x) noexcept
//	{
//		return -static_cast<Vector<float>>(x);
//	}
//	static inline Vector<bfloat16> operator*(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) * static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator/(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) / static_cast<Vector<float>>(rhs);
//	}
//
//	/*
//	 * Mixed precision arithmetics
//	 */
//	static inline Vector<bfloat16> operator+(Vector<bfloat16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) + rhs;
//	}
//	static inline Vector<bfloat16> operator+(Vector<float> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return lhs + static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator-(Vector<bfloat16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) - rhs;
//	}
//	static inline Vector<bfloat16> operator-(Vector<float> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return lhs - static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator*(Vector<bfloat16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) * rhs;
//	}
//	static inline Vector<bfloat16> operator*(Vector<float> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return lhs * static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<bfloat16> operator/(Vector<bfloat16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) / rhs;
//	}
//	static inline Vector<bfloat16> operator/(Vector<float> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return lhs / static_cast<Vector<float>>(rhs);
//	}
//
//	/**
//	 * result = (mask == 0xFFFFFFFF) ? x : y
//	 */
//	static inline Vector<bfloat16> select(Vector<bfloat16> mask, Vector<bfloat16> x, Vector<bfloat16> y)
//	{
//		return select(static_cast<Vector<float>>(mask), static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
//	}
//
//	/*
//	 * Fused multiply accumulate
//	 */
//
//	/* Calculates a * b + c */
//	static inline Vector<bfloat16> mul_add(Vector<bfloat16> a, Vector<bfloat16> b, Vector<bfloat16> c) noexcept
//	{
//		return Vector<bfloat16>(mul_add(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates a * b - c */
//	static inline Vector<bfloat16> mul_sub(Vector<bfloat16> a, Vector<bfloat16> b, Vector<bfloat16> c) noexcept
//	{
//		return Vector<bfloat16>(mul_sub(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates - a * b + c */
//	static inline Vector<bfloat16> neg_mul_add(Vector<bfloat16> a, Vector<bfloat16> b, Vector<bfloat16> c) noexcept
//	{
//		return Vector<bfloat16>(neg_mul_add(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates - a * b - c */
//	static inline Vector<bfloat16> neg_mul_sub(Vector<bfloat16> a, Vector<bfloat16> b, Vector<bfloat16> c) noexcept
//	{
//		return Vector<bfloat16>(neg_mul_sub(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//
//	/*
//	 * Float vector functions
//	 */
//	static inline Vector<bfloat16> max(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return max(static_cast<Vector<float>>(lhs), static_cast<Vector<float>>(rhs));
//	}
//	static inline Vector<bfloat16> min(Vector<bfloat16> lhs, Vector<bfloat16> rhs) noexcept
//	{
//		return min(static_cast<Vector<float>>(lhs), static_cast<Vector<float>>(rhs));
//	}
//	static inline Vector<bfloat16> abs(Vector<bfloat16> x) noexcept
//	{
//		return abs(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> sqrt(Vector<bfloat16> x) noexcept
//	{
//		return sqrt(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> rsqrt(Vector<bfloat16> x) noexcept
//	{
//		return rsqrt(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> rcp(Vector<bfloat16> x) noexcept
//	{
//		return rcp(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> sgn(Vector<bfloat16> x) noexcept
//	{
//		return sgn(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> floor(Vector<bfloat16> x) noexcept
//	{
//		return floor(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<bfloat16> ceil(Vector<bfloat16> x) noexcept
//	{
//		return ceil(static_cast<Vector<float>>(x));
//	}
//
//	/*
//	 * Horizontal functions
//	 */
//	static inline bfloat16 horizontal_add(Vector<bfloat16> x) noexcept
//	{
//		return scalar::float_to_bfloat16(horizontal_add(static_cast<Vector<float>>(x)));
//	}
//	static inline bfloat16 horizontal_mul(Vector<bfloat16> x) noexcept
//	{
//		return scalar::float_to_bfloat16(horizontal_mul(static_cast<Vector<float>>(x)));
//	}
//	static inline bfloat16 horizontal_min(Vector<bfloat16> x) noexcept
//	{
//		return scalar::float_to_bfloat16(horizontal_min(static_cast<Vector<float>>(x)));
//	}
//	static inline bfloat16 horizontal_max(Vector<bfloat16> x) noexcept
//	{
//		return scalar::float_to_bfloat16(horizontal_max(static_cast<Vector<float>>(x)));
//	}
//
//}

#endif /* VECTORS_BF16_VECTOR_HPP_ */
