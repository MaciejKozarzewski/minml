/*
 * fp32_ymm.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_YMM_HPP_
#define VECTORS_FP32_YMM_HPP_

#include "generic_vector.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
#if COMPILED_WITH_AVX

	template<>
	class Vector<float, YMM>
	{
		private:
			__m256 m_data;
			static inline __m256 broadcast(float x) noexcept
			{
				return _mm256_broadcast_ss(&x);
			}
		public:
			Vector() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			template<typename T>
			Vector(const T *src) noexcept // @suppress("Class members should be properly initialized")
			{
				load(src);
			}
			template<typename T>
			Vector(const T *src, int num) noexcept // @suppress("Class members should be properly initialized")
			{
				partial_load(src, num);
			}
			Vector(double x) noexcept :
					m_data(broadcast(static_cast<float>(x)))
			{
			}
			Vector(float x) noexcept :
					m_data(broadcast(x))
			{
			}
			Vector(float16 x) noexcept :
#if COMPILED_WITH_F16C
					m_data(broadcast(_cvtsh_ss(x.m_data)))
#else
					m_data(_mm256_setzero_ps())
#endif
			{
			}
			Vector(__m256i raw_bytes) noexcept :
					m_data(_mm256_castsi256_ps(raw_bytes))
			{
			}
			Vector(__m256 x) noexcept :
					m_data(x)
			{
			}
			Vector(__m128 low, __m128 high) noexcept :
					m_data(_mm256_setr_m128(low, high))
			{
			}
			Vector<float, YMM>& operator=(__m256 x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m256() const noexcept
			{
				return m_data;
			}
			__m128 low() const noexcept
			{
				return _mm256_castps256_ps128(m_data);
			}
			__m128 high() const noexcept
			{
				return _mm256_extractf128_ps(m_data, 1);
			}
			__m256i raw_bytes() const noexcept
			{
				return _mm256_castps_si256(m_data);
			}
			void load(const float *src) noexcept
			{
				assert(src != nullptr);
				m_data = _mm256_loadu_ps(src);
			}
			void partial_load(const float *src, int num) noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= size());
				m_data = _mm256_maskload_ps(src, get_mask(num));
			}
			void load(const float16 *src) noexcept
			{
#if COMPILED_WITH_F16C
				assert(src != nullptr);
				const __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
				m_data = _mm256_cvtph_ps(tmp);
#endif
			}
			void partial_load(const float16 *src, int num) noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= size());
				float16 tmp[8] = { float16(), float16(), float16(), float16(), float16(), float16(), float16(), float16() };
				for (int i = 0; i < num; i++)
					tmp[i] = src[i];
				load(tmp);
			}
			void store(float *dst) const noexcept
			{
				assert(dst != nullptr);
				_mm256_storeu_ps(dst, m_data);
			}
			void partial_store(float *dst, int num) const noexcept
			{
				assert(dst != nullptr);
				assert(0 <= num && num <= size());
				_mm256_maskstore_ps(dst, get_mask(num), m_data);
			}
			void store(float16 *dst) const noexcept
			{
#if COMPILED_WITH_F16C
				assert(dst != nullptr);
				const __m128i tmp = _mm256_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), tmp);
#endif
			}
			void partial_store(float16 *dst, int num) const noexcept
			{
				assert(dst != nullptr);
				assert(0 <= num && num <= size());
				float16 tmp[8];
				store(tmp);
				for (int i = 0; i < num; i++)
					dst[i] = tmp[i];
			}

			static Vector<float, YMM> zero() noexcept
			{
				return Vector<float, YMM>(0.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, YMM> one() noexcept
			{
				return Vector<float, YMM>(1.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, YMM> epsilon() noexcept
			{
				return Vector<float, YMM>(std::numeric_limits<float>::epsilon()); // @suppress("Ambiguous problem")
			}

			static constexpr int size() noexcept
			{
				return 8;
			}
			static constexpr RegisterType register_type() noexcept
			{
				return YMM;
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
	 */
	static inline Vector<float, YMM> operator==(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 0);
	}
	static inline Vector<float, YMM> operator!=(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 4);
	}
	static inline Vector<float, YMM> operator<(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 1);
	}
	static inline Vector<float, YMM> operator<=(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 2);
	}
	static inline Vector<float, YMM> operator>(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 14);
	}
	static inline Vector<float, YMM> operator>=(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 13);
	}
	static inline Vector<float, YMM> operator&(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_and_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> operator|(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_or_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> operator^(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_xor_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> operator~(Vector<float, YMM> x) noexcept
	{
#if COMPILED_WITH_AVX2
		const __m256i y = _mm256_castps_si256(x);
		const __m256i tmp = _mm256_cmpeq_epi32(y, y);
#else
		const __m256i tmp = ymm_constant<0xFFFFFFFFu>();
#endif
		return _mm256_xor_ps(x, _mm256_castsi256_ps(tmp));
	}
	static inline Vector<float, YMM> operator!(Vector<float, YMM> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline Vector<float, YMM> operator+(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_add_ps(lhs, rhs);
	}

	static inline Vector<float, YMM> operator-(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_sub_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> operator-(Vector<float, YMM> x) noexcept
	{
		return _mm256_xor_ps(x, Vector<float, YMM>(-0.0f)); // @suppress("Ambiguous problem")
	}

	static inline Vector<float, YMM> operator*(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_mul_ps(lhs, rhs);
	}

	static inline Vector<float, YMM> operator/(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm256_mul_ps(lhs, _mm256_rcp_ps(rhs));
#else
		return _mm256_div_ps(lhs, rhs);
#endif
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline Vector<float, YMM> select(Vector<float, YMM> mask, Vector<float, YMM> x, Vector<float, YMM> y)
	{
		return _mm256_blendv_ps(y, x, mask);
	}

	/* Float vector functions */
	static inline Vector<float, YMM> max(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_max_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> min(Vector<float, YMM> lhs, Vector<float, YMM> rhs) noexcept
	{
		return _mm256_min_ps(lhs, rhs);
	}
	static inline Vector<float, YMM> abs(Vector<float, YMM> x) noexcept
	{
		return _mm256_and_ps(x, _mm256_castsi256_ps(ymm_constant<0x7FFFFFFFu>()));
	}
	static inline Vector<float, YMM> sqrt(Vector<float, YMM> x) noexcept
	{
		return _mm256_sqrt_ps(x);
	}
	static inline Vector<float, YMM> rsqrt(Vector<float, YMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm256_rsqrt_ps(x);
#else
		return Vector<float, YMM>::one() / sqrt(x);
#endif
	}
	static inline Vector<float, YMM> rcp(Vector<float, YMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm256_rcp_ps(x);
#else
		return Vector<float, YMM>::one() / x;
#endif
	}
	static inline Vector<float, YMM> sgn(Vector<float, YMM> x) noexcept
	{
		__m256 zero = _mm256_setzero_ps();
		__m256 positive = _mm256_and_ps(_mm256_cmp_ps(zero, x, 1), _mm256_set1_ps(1.0f));
		__m256 negative = _mm256_and_ps(_mm256_cmp_ps(x, zero, 1), _mm256_set1_ps(-1.0f));
		return _mm256_or_ps(positive, negative);
	}
	static inline Vector<float, YMM> floor(Vector<float, YMM> x) noexcept
	{
		return _mm256_floor_ps(x);
	}
	static inline Vector<float, YMM> ceil(Vector<float, YMM> x) noexcept
	{
		return _mm256_ceil_ps(x);
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline Vector<float, YMM> mul_add(Vector<float, YMM> a, Vector<float, YMM> b, Vector<float, YMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fmadd_ps(a, b, c);
#else
		return a * b + c;
#endif
	}
	/* Calculates a * b - c */
	static inline Vector<float, YMM> mul_sub(Vector<float, YMM> a, Vector<float, YMM> b, Vector<float, YMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fmsub_ps(a, b, c);
#else
		return a * b - c;
#endif
	}
	/* Calculates - a * b + c */
	static inline Vector<float, YMM> neg_mul_add(Vector<float, YMM> a, Vector<float, YMM> b, Vector<float, YMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fnmadd_ps(a, b, c);
#else
		return -a * b + c;
#endif
	}
	/* Calculates - a * b - c */
	static inline Vector<float, YMM> neg_mul_sub(Vector<float, YMM> a, Vector<float, YMM> b, Vector<float, YMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fnmsub_ps(a, b, c);
#else
		return -a * b - c;
#endif
	}

	/*
	 * Horizontal functions
	 */
	static inline float horizontal_add(Vector<float, YMM> x) noexcept
	{
		const __m128 y = _mm_add_ps(x.low(), x.high());
		return horizontal_add(Vector<float, XMM>(y));
	}
	static inline float horizontal_mul(Vector<float, YMM> x) noexcept
	{
		const __m128 y = _mm_mul_ps(x.low(), x.high());
		return horizontal_mul(Vector<float, XMM>(y));
	}
	static inline float horizontal_min(Vector<float, YMM> x) noexcept
	{
		const __m128 y = _mm_min_ps(x.low(), x.high());
		return horizontal_min(Vector<float, XMM>(y));
	}
	static inline float horizontal_max(Vector<float, YMM> x) noexcept
	{
		const __m128 y = _mm_max_ps(x.low(), x.high());
		return horizontal_max(Vector<float, XMM>(y));
	}

#endif /* SUPPORTS_AVX */

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP32_YMM_HPP_ */
