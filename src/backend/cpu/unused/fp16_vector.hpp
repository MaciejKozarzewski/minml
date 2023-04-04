/*
 * fp16_vector.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP16_VECTOR_HPP_
#define VECTORS_FP16_VECTOR_HPP_

//#include "fp32_vector.hpp"
//#include "generic_vector.hpp"
//#include "vector_length.hpp"
//#include "vector_load_store.hpp"
//#include "vector_utils.hpp"
//
//namespace scalar
//{
//	static inline float float16_to_float(float16 x) noexcept
//	{
//#if SUPPORTS_FP16
//		return _cvtsh_ss(x.m_data);
//#else
//		return 0.0f;
//#endif
//	}
//	static inline float16 float_to_float16(float x) noexcept
//	{
//#if SUPPORTS_FP16
//		return float16 { _cvtss_sh(x, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)) };
//#else
//		return float16 { 0u };
//#endif
//	}
//}
//
//namespace SIMD_NAMESPACE
//{
//#if SUPPORTS_AVX
//
//	static inline __m256 float16_to_float(__m128i x) noexcept
//	{
//#  if SUPPORTS_FP16
//		return _mm256_cvtph_ps(x);
//#  else
//		return _mm256_setzero_ps();
//#  endif
//	}
//	static inline __m128i float_to_float16(__m256 x) noexcept
//	{
//#  if SUPPORTS_FP16
//		return _mm256_cvtps_ph(x, _MM_FROUND_NO_EXC);
//#  else
//		return _mm_setzero_si128();
//#  endif
//	}
//
//#elif SUPPORTS_SSE2 /* if __AVX__ is not defined */
//
//	static inline __m128 float16_to_float(__m128i x) noexcept
//	{
//#  if SUPPORTS_FP16
//		return _mm_cvtph_ps(x);
//#  else
//		return _mm_setzero_ps();
//#  endif
//	}
//	static inline __m128i float_to_float16(__m128 x) noexcept
//	{
//#  if SUPPORTS_FP16
//		return _mm_cvtps_ph(x, _MM_FROUND_NO_EXC);
//#  else
//		return _mm_setzero_si128();
//#  endif
//	}
//
//#else /* if __SSE2__ is not defined */
//
//	static inline float float16_to_float(float16 x) noexcept
//	{
//		return scalar::float16_to_float(x);
//	}
//	static inline float16 float_to_float16(float x) noexcept
//	{
//		return scalar::float_to_float16(x);
//	}
//
//#endif
//
//	template<>
//	class Vector<float16>
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
//			static constexpr int length = vector_length<float16>();
//
//			Vector() noexcept // @suppress("Class members should be properly initialized")
//			{
//			}
//			Vector(const float *ptr, int num = length) noexcept :
//					m_data(vector_load(ptr, num))
//			{
//			}
//			Vector(const float16 *ptr, int num = length) noexcept
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
//			Vector(float16 x) noexcept :
//					Vector(scalar::float16_to_float(x))
//			{
//			}
//			Vector(double x) noexcept :
//					Vector(static_cast<float>(x))
//			{
//			}
//			operator Vector<float>() const noexcept
//			{
//				return Vector<float>(m_data); // @suppress("Ambiguous problem")
//			}
//			void load(const float *ptr, int num = length) noexcept
//			{
//				m_data = vector_load(ptr, num);
//			}
//			void load(const float16 *ptr, int num = length) noexcept
//			{
//			}
//			void store(float *ptr, int num = length) const noexcept
//			{
//				vector_store(m_data, ptr, num);
//			}
//			void store(float16 *ptr, int num = length) const noexcept
//			{
//			}
//			void insert(float value, int index) noexcept
//			{
////				m_data.insert(value, index);  // FIXME
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
//			void cutoff(const int num, Vector<float16> value = zero()) noexcept
//			{
//				m_data = cutoff_ps(m_data, num, value.m_data);
//			}
//
//			static constexpr float16 scalar_zero() noexcept
//			{
//				return float16 { 0x0000u };
//			}
//			static constexpr float16 scalar_one() noexcept
//			{
//				return float16 { 0x3c00u };
//			}
//			static constexpr float16 scalar_epsilon() noexcept
//			{
//				return float16 { 0x0400u };
//			}
//
//			static Vector<float16> zero() noexcept
//			{
//				return Vector<float16>(scalar_zero());
//			}
//			static Vector<float16> one() noexcept
//			{
//				return Vector<float16>(scalar_one());
//			}
//			static Vector<float16> epsilon() noexcept
//			{
//				return Vector<float16>(scalar_epsilon());
//			}
//	};
//
//	/*
//	 * Float vector logical operations.
//	 * Return vector of half floats, either 0x0000 (0.0f) for false, or 0xFFFF (-nan) for true.
//	 */
//	static inline Vector<float16> operator==(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) == static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator!=(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) != static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator<(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) < static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator<=(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) <= static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator>(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) > static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator>=(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) >= static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator&(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) & static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator|(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) | static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator^(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) ^ static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator~(Vector<float16> x) noexcept
//	{
//		return ~static_cast<Vector<float>>(x);
//	}
//	static inline Vector<float16> operator!(Vector<float16> x) noexcept
//	{
//		return ~x;
//	}
//
//	/*
//	 * Float vector arithmetics.
//	 */
//	static inline Vector<float16> operator+(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) + static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator-(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) - static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator-(Vector<float16> x) noexcept
//	{
//		return -static_cast<Vector<float>>(x);
//	}
//	static inline Vector<float16> operator*(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) * static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator/(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) / static_cast<Vector<float>>(rhs);
//	}
//
//	/*
//	 * Mixed precision arithmetics
//	 */
//	static inline Vector<float16> operator+(Vector<float16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) + rhs;
//	}
//	static inline Vector<float16> operator+(Vector<float> lhs, Vector<float16> rhs) noexcept
//	{
//		return lhs + static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator-(Vector<float16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) - rhs;
//	}
//	static inline Vector<float16> operator-(Vector<float> lhs, Vector<float16> rhs) noexcept
//	{
//		return lhs - static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator*(Vector<float16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) * rhs;
//	}
//	static inline Vector<float16> operator*(Vector<float> lhs, Vector<float16> rhs) noexcept
//	{
//		return lhs * static_cast<Vector<float>>(rhs);
//	}
//	static inline Vector<float16> operator/(Vector<float16> lhs, Vector<float> rhs) noexcept
//	{
//		return static_cast<Vector<float>>(lhs) / rhs;
//	}
//	static inline Vector<float16> operator/(Vector<float> lhs, Vector<float16> rhs) noexcept
//	{
//		return lhs / static_cast<Vector<float>>(rhs);
//	}
//
//	/**
//	 * result = (mask == 0xFFFFFFFF) ? x : y
//	 */
//	static inline Vector<float16> select(Vector<float16> mask, Vector<float16> x, Vector<float16> y)
//	{
//		return select(static_cast<Vector<float>>(mask), static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
//	}
//
//	/*
//	 * Float vector functions
//	 */
//	static inline Vector<float16> max(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return max(static_cast<Vector<float>>(lhs), static_cast<Vector<float>>(rhs));
//	}
//	static inline Vector<float16> min(Vector<float16> lhs, Vector<float16> rhs) noexcept
//	{
//		return min(static_cast<Vector<float>>(lhs), static_cast<Vector<float>>(rhs));
//	}
//	static inline Vector<float16> abs(Vector<float16> x) noexcept
//	{
//		return abs(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> sqrt(Vector<float16> x) noexcept
//	{
//		return sqrt(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> rsqrt(Vector<float16> x) noexcept
//	{
//		return rsqrt(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> rcp(Vector<float16> x) noexcept
//	{
//		return rcp(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> sgn(Vector<float16> x) noexcept
//	{
//		return sgn(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> floor(Vector<float16> x) noexcept
//	{
//		return floor(static_cast<Vector<float>>(x));
//	}
//	static inline Vector<float16> ceil(Vector<float16> x) noexcept
//	{
//		return ceil(static_cast<Vector<float>>(x));
//	}
//
//	/*
//	 * Fused multiply accumulate
//	 */
//
//	/* Calculates a * b + c */
//	static inline Vector<float16> mul_add(Vector<float16> a, Vector<float16> b, Vector<float16> c) noexcept
//	{
//		return Vector<float16>(mul_add(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates a * b - c */
//	static inline Vector<float16> mul_sub(Vector<float16> a, Vector<float16> b, Vector<float16> c) noexcept
//	{
//		return Vector<float16>(mul_sub(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates - a * b + c */
//	static inline Vector<float16> neg_mul_add(Vector<float16> a, Vector<float16> b, Vector<float16> c) noexcept
//	{
//		return Vector<float16>(neg_mul_add(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//	/* Calculates - a * b - c */
//	static inline Vector<float16> neg_mul_sub(Vector<float16> a, Vector<float16> b, Vector<float16> c) noexcept
//	{
//		return Vector<float16>(neg_mul_sub(static_cast<Vector<float>>(a), static_cast<Vector<float>>(b), static_cast<Vector<float>>(c)));
//	}
//
//	/*
//	 * Horizontal functions
//	 */
//	static inline float16 horizontal_add(Vector<float16> x) noexcept
//	{
//		return scalar::float_to_float16(horizontal_add(static_cast<Vector<float>>(x)));
//	}
//	static inline float16 horizontal_mul(Vector<float16> x) noexcept
//	{
//		return scalar::float_to_float16(horizontal_mul(static_cast<Vector<float>>(x)));
//	}
//	static inline float16 horizontal_min(Vector<float16> x) noexcept
//	{
//		return scalar::float_to_float16(horizontal_min(static_cast<Vector<float>>(x)));
//	}
//	static inline float16 horizontal_max(Vector<float16> x) noexcept
//	{
//		return scalar::float_to_float16(horizontal_max(static_cast<Vector<float>>(x)));
//	}
//}

#endif /* VECTORS_FP16_VECTOR_HPP_ */
