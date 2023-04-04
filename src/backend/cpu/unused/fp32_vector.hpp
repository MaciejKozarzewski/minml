/*
 * fp32_vector.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_VECTOR_HPP_
#define VECTORS_FP32_VECTOR_HPP_

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

//#include "generic_vector.hpp"
//#include "vector_load.hpp"
//#include "vector_store.hpp"
//#include "vector_utils.hpp"
//#include "types.hpp"
//#include "type_conversions.hpp"

namespace SIMD_NAMESPACE
{
//	namespace internal
//	{
//#if SUPPORTS_AVX
//		static inline __m256 broadcast(float x) noexcept
//		{
//			return _mm256_broadcast_ss(&x);
//		}
//#elif SUPPORTS_SSE2
//		static inline __m128 broadcast(float x) noexcept
//		{
//			return _mm_set1_ps(x);
//		}
//#else
//		static inline float broadcast(float x) noexcept
//		{
//			return x;
//		}
//#endif
//	}

//	template<>
//	class Vector<float>
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
//			static constexpr int length = vector_length<float>();
//
//			Vector() noexcept // @suppress("Class members should be properly initialized")
//			{
//			}
//			template<typename T>
//			Vector(const T *ptr, int num = length) noexcept // @suppress("Class members should be properly initialized")
//			{
//				load(ptr, num);
//			}
//			Vector(double x) noexcept :
//					m_data(internal::broadcast(static_cast<float>(x)))
//			{
//			}
//			Vector(float x) noexcept :
//					m_data(internal::broadcast(x))
//			{
//			}
//			Vector(float16 x) noexcept :
//					m_data(internal::broadcast(Converter<RegisterType::SCALAR, float16, float>()(x)))
//			{
//			}
//			Vector(sw_float16 x) noexcept :
//					m_data(internal::broadcast(Converter<RegisterType::SCALAR, sw_float16, float>()(x)))
//			{
//			}
//			Vector(bfloat16 x) noexcept :
//					m_data(internal::broadcast(Converter<RegisterType::SCALAR, bfloat16, float>()(x)))
//			{
//			}
//			Vector(sw_bfloat16 x) noexcept :
//					m_data(internal::broadcast(Converter<RegisterType::SCALAR, sw_bfloat16, float>()(x)))
//			{
//			}
//#if SUPPORTS_AVX
//			Vector(__m256i raw_bytes) noexcept :
//					m_data(_mm256_castsi256_ps(raw_bytes))
//			{
//			}
//			Vector(__m256 x) noexcept :
//					m_data(x)
//			{
//			}
//			Vector(__m128 low) noexcept :
//					m_data(combine(low, _mm_setzero_ps()))
//			{
//			}
//			Vector(__m128 low, __m128 high) noexcept :
//					m_data(combine(low, high))
//			{
//			}
//			Vector<float>& operator=(__m256 x) noexcept
//			{
//				m_data = x;
//				return *this;
//			}
//			operator __m256() const noexcept
//			{
//				return m_data;
//			}
//#elif SUPPORTS_SSE2
//			Vector(__m128i raw_bytes) noexcept :
//					m_data(_mm_castsi128_ps(raw_bytes))
//			{
//			}
//			Vector(__m128 x) noexcept :
//					m_data(x)
//			{
//			}
//			Vector<float>& operator=(__m128 x) noexcept
//			{
//				m_data = x;
//				return *this;
//			}
//			operator __m128() const noexcept
//			{
//				return m_data;
//			}
//#else
//			Vector(uint32_t raw_bytes) noexcept :
//					m_data(bitwise_cast<float>(raw_bytes))
//			{
//			}
//			operator float() const noexcept
//			{
//				return m_data;
//			}
//#endif
//			template<typename T>
//			void load(const T *ptr, int num = length) noexcept
//			{
//				const auto tmp = Loader<RegisterType::AUTO>()(ptr, num);
//				m_data = Converter<RegisterType::AUTO, T, float>()(tmp);
//			}
//			template<typename T>
//			void store(T *ptr, int num = length) const noexcept
//			{
//				const auto tmp = Converter<RegisterType::AUTO, float, T>()(m_data);
//				Storer<RegisterType::AUTO>()(ptr, tmp, num);
//			}
//			template<typename T>
//			void insert(T value, int index) noexcept
//			{
//				assert(index >= 0 && index < length);
//				const float x = Converter<RegisterType::SCALAR, T, float>()(value);
//#if SUPPORTS_AVX
//				__m256 tmp = internal::broadcast(x);
//				switch (index)
//				{
//					case 0:
//						m_data = _mm256_blend_ps(m_data, tmp, 1);
//						break;
//					case 1:
//						m_data = _mm256_blend_ps(m_data, tmp, 2);
//						break;
//					case 2:
//						m_data = _mm256_blend_ps(m_data, tmp, 4);
//						break;
//					case 3:
//						m_data = _mm256_blend_ps(m_data, tmp, 8);
//						break;
//					case 4:
//						m_data = _mm256_blend_ps(m_data, tmp, 16);
//						break;
//					case 5:
//						m_data = _mm256_blend_ps(m_data, tmp, 32);
//						break;
//					case 6:
//						m_data = _mm256_blend_ps(m_data, tmp, 64);
//						break;
//					default:
//						m_data = _mm256_blend_ps(m_data, tmp, 128);
//						break;
//				}
//#elif SUPPORTS_SSE2
//				float tmp[4];
//				store(tmp);
//				tmp[index] = value;
//				load(tmp);
//#else
//				m_data = value;
//#endif
//			}
//			float extract(int index) const noexcept
//			{
//				assert(index >= 0 && index < length);
//				float tmp[length];
//				store(tmp);
//				return tmp[index];
//			}
//			float operator[](int index) const noexcept
//			{
//				return extract(index);
//			}
//			void cutoff(const int num, Vector<float> value = zero()) noexcept
//			{
//				m_data = cutoff_ps(m_data, num, value.m_data);
//			}
//
//			static constexpr float scalar_zero() noexcept
//			{
//				return 0.0f;
//			}
//			static constexpr float scalar_one() noexcept
//			{
//				return 1.0f;
//			}
//			static constexpr float scalar_epsilon() noexcept
//			{
//				return std::numeric_limits<float>::epsilon();
//			}
//
//			static Vector<float> zero() noexcept
//			{
//				return Vector<float>(scalar_zero()); // @suppress("Ambiguous problem")
//			}
//			static Vector<float> one() noexcept
//			{
//				return Vector<float>(scalar_one()); // @suppress("Ambiguous problem")
//			}
//			static Vector<float> epsilon() noexcept
//			{
//				return Vector<float>(scalar_epsilon()); // @suppress("Ambiguous problem")
//			}
//	};
//
//	/*
//	 * Float vector logical operations.
//	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
//	 */
//	static inline Vector<float> operator==(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 0);
//#elif SUPPORTS_SSE2
//		return _mm_cmpeq_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) == static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator!=(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 4);
//#elif SUPPORTS_SSE2
//		return _mm_cmpneq_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) != static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator<(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 1);
//#elif SUPPORTS_SSE2
//		return _mm_cmplt_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) < static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator<=(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 2);
//#elif SUPPORTS_SSE2
//		return _mm_cmple_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) <= static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator>(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 14);
//#elif SUPPORTS_SSE2
//		return _mm_cmpgt_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) > static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator>=(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_cmp_ps(lhs, rhs, 13);
//#elif SUPPORTS_SSE2
//		return _mm_cmpge_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(static_cast<float>(lhs) >= static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
//#endif
//	}
//	static inline Vector<float> operator&(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_and_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_and_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(bitwise_cast<uint32_t>(static_cast<float>(lhs)) & bitwise_cast<uint32_t>(static_cast<float>(rhs)));
//#endif
//	}
//	static inline Vector<float> operator|(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_or_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_or_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(bitwise_cast<uint32_t>(static_cast<float>(lhs)) | bitwise_cast<uint32_t>(static_cast<float>(rhs)));
//#endif
//	}
//	static inline Vector<float> operator^(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_xor_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_xor_ps(lhs, rhs);
//#else
//		return bitwise_cast<float>(bitwise_cast<uint32_t>(static_cast<float>(lhs)) ^ bitwise_cast<uint32_t>(static_cast<float>(rhs)));
//#endif
//	}
//	static inline Vector<float> operator~(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_xor_ps(x, _mm256_castsi256_ps(constant<0xFFFFFFFFu>()));
//#elif SUPPORTS_SSE2
//		return _mm_xor_ps(x, _mm_castsi128_ps(constant<0xFFFFFFFFu>()));
//#else
//		return bitwise_cast<float>(~bitwise_cast<uint32_t>(static_cast<float>(x)));
//#endif
//	}
//	static inline Vector<float> operator!(Vector<float> x) noexcept
//	{
//		return ~x;
//	}
//
//	/*
//	 * Float vector arithmetics.
//	 */
//	static inline Vector<float> operator+(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_add_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_add_ps(lhs, rhs);
//#else
//		return static_cast<float>(lhs) + static_cast<float>(rhs);
//#endif
//	}
//
//	static inline Vector<float> operator-(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_sub_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_sub_ps(lhs, rhs);
//#else
//		return static_cast<float>(lhs) - static_cast<float>(rhs);
//#endif
//	}
//	static inline Vector<float> operator-(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_xor_ps(x, Vector<float>(-0.0f)); // @suppress("Ambiguous problem")
//#elif SUPPORTS_SSE2
//		return _mm_xor_ps(x, Vector<float>(-0.0f)); // @suppress("Ambiguous problem")
//#else
//		return -static_cast<float>(x);
//#endif
//	}
//
//	static inline Vector<float> operator*(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_mul_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_mul_ps(lhs, rhs);
//#else
//		return static_cast<float>(lhs) * static_cast<float>(rhs);
//#endif
//	}
//
//	static inline Vector<float> operator/(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if ENABLE_FAST_MATH
//#  if SUPPORTS_AVX
//		return _mm256_mul_ps(lhs, _mm256_rcp_ps(rhs));
//#  elif SUPPORTS_SSE2
//		return _mm_mul_ps(lhs, _mm_rcp_ps(rhs));
//#  else
//		return static_cast<float>(lhs) / static_cast<float>(rhs);
//#  endif
//#else
//# if SUPPORTS_AVX
//		return _mm256_div_ps(lhs, rhs);
//# elif SUPPORTS_SSE2
//		return _mm_div_ps(lhs, rhs);
//# else
//		return static_cast<float>(lhs) / static_cast<float>(rhs);
//# endif
//#endif
//	}
//
//	/**
//	 * result = (mask == 0xFFFFFFFF) ? x : y
//	 */
//	static inline Vector<float> select(Vector<float> mask, Vector<float> x, Vector<float> y)
//	{
//#if SUPPORTS_AVX
//		return _mm256_blendv_ps(y, x, mask);
//#elif SUPPORTS_SSE41
//		return _mm_blendv_ps(y, x, mask);
//#elif SUPPORTS_SSE2
//		return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, y));
//#else
//		return (bitwise_cast<uint32_t>(static_cast<float>(mask)) == 0xFFFFFFFFu) ? x : y;
//#endif
//	}
//
//	/* Float vector functions */
//	static inline Vector<float> max(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_max_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_max_ps(lhs, rhs);
//#else
//		return std::max(static_cast<float>(lhs), static_cast<float>(rhs));
//#endif
//	}
//	static inline Vector<float> min(Vector<float> lhs, Vector<float> rhs) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_min_ps(lhs, rhs);
//#elif SUPPORTS_SSE2
//		return _mm_min_ps(lhs, rhs);
//#else
//		return std::min(static_cast<float>(lhs), static_cast<float>(rhs));
//#endif
//	}
//	static inline Vector<float> abs(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_and_ps(x, _mm256_castsi256_ps(constant<0x7FFFFFFFu>()));
//#elif SUPPORTS_SSE2
//		return _mm_and_ps(x, _mm_castsi128_ps(constant<0x7FFFFFFFu>()));
//#else
//		return std::fabs(static_cast<float>(x));
//#endif
//	}
//	static inline Vector<float> sqrt(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_sqrt_ps(x);
//#elif SUPPORTS_SSE2
//		return _mm_sqrt_ps(x);
//#else
//		return std::sqrt(static_cast<float>(x));
//#endif
//	}
//	static inline Vector<float> rsqrt(Vector<float> x) noexcept
//	{
//#if ENABLE_FAST_MATH
//#  if SUPPORTS_AVX
//		return _mm256_rsqrt_ps(x);
//#  elif SUPPORTS_SSE2
//		return _mm_rsqrt_ps(x);
//#  else
//		return 1.0f / std::sqrt(static_cast<float>(x));
//#  endif
//#else
//		return Vector<float>::one() / sqrt(x);
//#endif
//	}
//	static inline Vector<float> rcp(Vector<float> x) noexcept
//	{
//#if ENABLE_FAST_MATH
//#  if SUPPORTS_AVX
//		return _mm256_rcp_ps(x);
//#  elif SUPPORTS_SSE2
//		return _mm_rcp_ps(x);
//#  else
//		return Vector<float>::one() / static_cast<float>(x);
//#  endif
//#else
//		return Vector<float>::one() / x;
//#endif
//	}
//	static inline Vector<float> sgn(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		__m256 zero = _mm256_setzero_ps();
//		__m256 positive = _mm256_and_ps(_mm256_cmp_ps(zero, x, 1), _mm256_set1_ps(1.0f));
//		__m256 negative = _mm256_and_ps(_mm256_cmp_ps(x, zero, 1), _mm256_set1_ps(-1.0f));
//		return _mm256_or_ps(positive, negative);
//#elif SUPPORTS_SSE2
//		__m128 zero = _mm_setzero_ps();
//		__m128 positive = _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_set1_ps(1.0f));
//		__m128 negative = _mm_and_ps(_mm_cmplt_ps(x, zero), _mm_set1_ps(-1.0f));
//		return _mm_or_ps(positive, negative);
//#else
//		return static_cast<float>((static_cast<float>(x) > 0.0f) - (static_cast<float>(x) < 0.0f));
//#endif
//	}
//	static inline Vector<float> floor(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_floor_ps(x);
//#elif SUPPORTS_SSE41
//		return _mm_floor_ps(x);
//#elif SUPPORTS_SSE2
//		float tmp[4];
//		x.store(tmp);
//		for (int i = 0; i < 4; i++)
//			tmp[i] = std::floor(tmp[i]);
//		return Vector<float>(tmp);
//#else
//		return std::floor(static_cast<float>(x));
//#endif
//	}
//	static inline Vector<float> ceil(Vector<float> x) noexcept
//	{
//#if SUPPORTS_AVX
//		return _mm256_ceil_ps(x);
//#elif SUPPORTS_SSE41
//		return _mm_ceil_ps(x);
//#elif SUPPORTS_SSE2
//		float tmp[4];
//		x.store(tmp);
//		for (int i = 0; i < 4; i++)
//			tmp[i] = std::ceil(tmp[i]);
//		return Vector<float>(tmp);
//#else
//		return std::ceil(static_cast<float>(x));
//#endif
//	}
//
//	/*
//	 * Fused multiply accumulate
//	 */
//
//	/* Calculates a * b + c */
//	static inline Vector<float> mul_add(Vector<float> a, Vector<float> b, Vector<float> c) noexcept
//	{
//#if SUPPORTS_AVX and SUPPORTS_FMA
//		return _mm256_fmadd_ps(a, b, c);
//#else
//		return a * b + c;
//#endif
//	}
//	/* Calculates a * b - c */
//	static inline Vector<float> mul_sub(Vector<float> a, Vector<float> b, Vector<float> c) noexcept
//	{
//#if SUPPORTS_AVX and SUPPORTS_FMA
//		return _mm256_fmsub_ps(a, b, c);
//#else
//		return a * b - c;
//#endif
//	}
//	/* Calculates - a * b + c */
//	static inline Vector<float> neg_mul_add(Vector<float> a, Vector<float> b, Vector<float> c) noexcept
//	{
//#if SUPPORTS_AVX and SUPPORTS_FMA
//		return _mm256_fnmadd_ps(a, b, c);
//#else
//		return -a * b + c;
//#endif
//	}
//	/* Calculates - a * b - c */
//	static inline Vector<float> neg_mul_sub(Vector<float> a, Vector<float> b, Vector<float> c) noexcept
//	{
//#if SUPPORTS_AVX and SUPPORTS_FMA
//		return _mm256_fnmsub_ps(a, b, c);
//#else
//		return -a * b - c;
//#endif
//	}
//
//	/*
//	 * Horizontal functions
//	 */
//
//	static inline float horizontal_add(Vector<float> x) noexcept
//	{
//#if SUPPORTS_SSE2
//#  if SUPPORTS_AVX
//		__m128 y = _mm_add_ps(get_low(x), get_high(x));
//#  else
//		__m128 y = x;
//#  endif
//		__m128 t1 = _mm_movehl_ps(y, y);
//		__m128 t2 = _mm_add_ps(y, t1);
//		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
//		__m128 t4 = _mm_add_ss(t2, t3);
//		return _mm_cvtss_f32(t4);
//#else
//		return static_cast<float>(x);
//#endif
//	}
//	static inline float horizontal_mul(Vector<float> x) noexcept
//	{
//#if SUPPORTS_SSE2
//#  if SUPPORTS_AVX
//		__m128 y = _mm_mul_ps(get_low(x), get_high(x));
//#  else
//		__m128 y = x;
//#  endif
//		__m128 t1 = _mm_movehl_ps(y, y);
//		__m128 t2 = _mm_mul_ps(y, t1);
//		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
//		__m128 t4 = _mm_mul_ps(t2, t3);
//		return _mm_cvtss_f32(t4);
//#else
//		return static_cast<float>(x);
//#endif
//	}
//	static inline float horizontal_min(Vector<float> x) noexcept
//	{
//#if SUPPORTS_SSE2
//#  if SUPPORTS_AVX
//		__m128 y = _mm_min_ps(get_low(x), get_high(x));
//#  else
//		__m128 y = x;
//#  endif
//		__m128 t1 = _mm_movehl_ps(y, y);
//		__m128 t2 = _mm_min_ps(y, t1);
//		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
//		__m128 t4 = _mm_min_ps(t2, t3);
//		return _mm_cvtss_f32(t4);
//#else
//		return static_cast<float>(x);
//#endif
//	}
//	static inline float horizontal_max(Vector<float> x) noexcept
//	{
//#if SUPPORTS_SSE2
//#  if SUPPORTS_AVX
//		__m128 y = _mm_max_ps(get_low(x), get_high(x));
//#  else
//		__m128 y = x;
//#  endif
//		__m128 t1 = _mm_movehl_ps(y, y);
//		__m128 t2 = _mm_max_ps(y, t1);
//		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
//		__m128 t4 = _mm_max_ps(t2, t3);
//		return _mm_cvtss_f32(t4);
//#else
//		return static_cast<float>(x);
//#endif
//	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP32_VECTOR_HPP_ */
