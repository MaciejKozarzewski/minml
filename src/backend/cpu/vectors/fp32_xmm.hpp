/*
 * fp32_xmm.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_XMM_HPP_
#define VECTORS_FP32_XMM_HPP_

#include "generic_vector.hpp"
#include "types.hpp"
#include "vector_constants.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
#if COMPILED_WITH_SSE2
	template<>
	class Vector<float, XMM>
	{
		private:
			__m128 m_data;
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
					m_data(_mm_set1_ps(static_cast<float>(x)))
			{
			}
			Vector(float x) noexcept :
					m_data(_mm_set1_ps(x))
			{
			}
			Vector(float16 x) noexcept :
#if COMPILED_WITH_F16C
					m_data(_mm_set1_ps(_cvtsh_ss(x.m_data)))
#else
					m_data(_mm_setzero_ps())
#endif
			{
			}
			Vector(__m128i raw_bytes) noexcept :
					m_data(_mm_castsi128_ps(raw_bytes))
			{
			}
			Vector(__m128 x) noexcept :
					m_data(x)
			{
			}
			Vector<float, XMM>& operator=(__m128 x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m128() const noexcept
			{
				return m_data;
			}
			void load(const float *src) noexcept
			{
				m_data = _mm_loadu_ps(src);
			}
			void partial_load(const float *src, int num) noexcept
			{
				assert(0 <= num && num <= size());
				float tmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				for (int i = 0; i < num; i++)
					tmp[i] = src[i];
				load(tmp);
			}
			void load(const float16 *src) noexcept
			{
#if COMPILED_WITH_F16C
				assert(src != nullptr);
				m_data = _mm_cvtph_ps(_mm_loadu_si64(src));
#endif
			}
			void partial_load(const float16 *src, int num) noexcept
			{
				assert(0 <= num && num <= size());
				float16 tmp[4] = { float16(), float16(), float16(), float16() };
				for (int i = 0; i < num; i++)
					tmp[i] = src[i];
				load(tmp);
			}
			void store(float *dst) const noexcept
			{
				_mm_storeu_ps(dst, m_data);
			}
			void partial_store(float *dst, int num) const noexcept
			{
				assert(0 <= num && num <= size());
				float tmp[4];
				store(tmp);
				for (int i = 0; i < num; i++)
					dst[i] = tmp[i];
			}
			void store(float16 *dst) const noexcept
			{
#if COMPILED_WITH_F16C
				assert(dst != nullptr);
				_mm_storeu_si64(dst, _mm_cvtps_ph(m_data, _MM_FROUND_NO_EXC));
#endif
			}
			void partial_store(float16 *dst, int num) const noexcept
			{
				assert(0 <= num && num <= size());
				float16 tmp[4];
				store(tmp);
				for (int i = 0; i < num; i++)
					dst[i] = tmp[i];
			}

			static Vector<float, XMM> zero() noexcept
			{
				return Vector<float, XMM>(0.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, XMM> one() noexcept
			{
				return Vector<float, XMM>(1.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, XMM> epsilon() noexcept
			{
				return Vector<float, XMM>(std::numeric_limits<float>::epsilon()); // @suppress("Ambiguous problem")
			}

			static constexpr int size() noexcept
			{
				return 4;
			}
			static constexpr RegisterType register_type() noexcept
			{
				return XMM;
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
	 */
	static inline Vector<float, XMM> operator==(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmpeq_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator!=(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmpneq_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator<(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmplt_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator<=(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmple_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator>(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmpgt_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator>=(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_cmpge_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator&(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_and_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator|(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_or_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator^(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_xor_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator~(Vector<float, XMM> x) noexcept
	{
		const __m128i y = _mm_castps_si128(x);
		const __m128i tmp = _mm_cmpeq_epi32(y, y);
		return _mm_xor_ps(x, _mm_castsi128_ps(tmp));
	}
	static inline Vector<float, XMM> operator!(Vector<float, XMM> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline Vector<float, XMM> operator+(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_add_ps(lhs, rhs);
	}

	static inline Vector<float, XMM> operator-(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_sub_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> operator-(Vector<float, XMM> x) noexcept
	{
		return _mm_xor_ps(x, Vector<float, XMM>(-0.0f)); // @suppress("Ambiguous problem")
	}

	static inline Vector<float, XMM> operator*(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_mul_ps(lhs, rhs);
	}

	static inline Vector<float, XMM> operator/(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm_mul_ps(lhs, _mm_rcp_ps(rhs));
#else
		return _mm_div_ps(lhs, rhs);
#endif
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline Vector<float, XMM> select(Vector<float, XMM> mask, Vector<float, XMM> x, Vector<float, XMM> y)
	{
#if SUPPORTS_SSE41
		return _mm_blendv_ps(y, x, mask);
#else
		return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, y));
#endif
	}

	/* Float vector functions */
	static inline Vector<float, XMM> max(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_max_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> min(Vector<float, XMM> lhs, Vector<float, XMM> rhs) noexcept
	{
		return _mm_min_ps(lhs, rhs);
	}
	static inline Vector<float, XMM> abs(Vector<float, XMM> x) noexcept
	{
		return _mm_and_ps(x, _mm_castsi128_ps(xmm_constant<0x7FFFFFFFu>()));
	}
	static inline Vector<float, XMM> sqrt(Vector<float, XMM> x) noexcept
	{
		return _mm_sqrt_ps(x);
	}
	static inline Vector<float, XMM> rsqrt(Vector<float, XMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm_rsqrt_ps(x);
#else
		return Vector<float, XMM>::one() / sqrt(x);
#endif
	}
	static inline Vector<float, XMM> rcp(Vector<float, XMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm_rcp_ps(x);
#else
		return Vector<float, XMM>::one() / x;
#endif
	}
	static inline Vector<float, XMM> sgn(Vector<float, XMM> x) noexcept
	{
		__m128 zero = _mm_setzero_ps();
		__m128 positive = _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_set1_ps(1.0f));
		__m128 negative = _mm_and_ps(_mm_cmplt_ps(x, zero), _mm_set1_ps(-1.0f));
		return _mm_or_ps(positive, negative);
	}
	static inline Vector<float, XMM> floor(Vector<float, XMM> x) noexcept
	{
#if COMPILED_WITH_SSE41
		return _mm_floor_ps(x);
#else
		float tmp[4];
		x.store(tmp);
		for (int i = 0; i < 4; i++)
			tmp[i] = std::floor(tmp[i]);
		return Vector<float, XMM>(tmp);
#endif
	}
	static inline Vector<float, XMM> ceil(Vector<float, XMM> x) noexcept
	{
#if COMPILED_WITH_SSE41
		return _mm_ceil_ps(x);
#else
		float tmp[4];
		x.store(tmp);
		for (int i = 0; i < 4; i++)
			tmp[i] = std::ceil(tmp[i]);
		return Vector<float, XMM>(tmp);
#endif
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline Vector<float, XMM> mul_add(Vector<float, XMM> a, Vector<float, XMM> b, Vector<float, XMM> c) noexcept
	{
		return a * b + c;
	}
	/* Calculates a * b - c */
	static inline Vector<float, XMM> mul_sub(Vector<float, XMM> a, Vector<float, XMM> b, Vector<float, XMM> c) noexcept
	{
		return a * b - c;
	}
	/* Calculates - a * b + c */
	static inline Vector<float, XMM> neg_mul_add(Vector<float, XMM> a, Vector<float, XMM> b, Vector<float, XMM> c) noexcept
	{
		return -a * b + c;
	}
	/* Calculates - a * b - c */
	static inline Vector<float, XMM> neg_mul_sub(Vector<float, XMM> a, Vector<float, XMM> b, Vector<float, XMM> c) noexcept
	{
		return -a * b - c;
	}

	/*
	 * Horizontal functions
	 */
	static inline float horizontal_add(Vector<float, XMM> x) noexcept
	{
		__m128 t1 = _mm_movehl_ps(x, x);
		__m128 t2 = _mm_add_ps(x, t1);
		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
		__m128 t4 = _mm_add_ss(t2, t3);
		return _mm_cvtss_f32(t4);
	}
	static inline float horizontal_mul(Vector<float, XMM> x) noexcept
	{
		__m128 t1 = _mm_movehl_ps(x, x);
		__m128 t2 = _mm_mul_ps(x, t1);
		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
		__m128 t4 = _mm_mul_ps(t2, t3);
		return _mm_cvtss_f32(t4);
	}
	static inline float horizontal_min(Vector<float, XMM> x) noexcept
	{
		__m128 t1 = _mm_movehl_ps(x, x);
		__m128 t2 = _mm_min_ps(x, t1);
		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
		__m128 t4 = _mm_min_ps(t2, t3);
		return _mm_cvtss_f32(t4);
	}
	static inline float horizontal_max(Vector<float, XMM> x) noexcept
	{
		__m128 t1 = _mm_movehl_ps(x, x);
		__m128 t2 = _mm_max_ps(x, t1);
		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
		__m128 t4 = _mm_max_ps(t2, t3);
		return _mm_cvtss_f32(t4);
	}

#endif /* SUPPORTS_SSE2 */

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP32_XMM_HPP_ */
