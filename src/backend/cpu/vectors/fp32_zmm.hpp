/*
 * fp32_zmm.hpp
 *
 *  Created on: Apr 15, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_VECTORS_FP32_ZMM_HPP_
#define BACKEND_CPU_VECTORS_FP32_ZMM_HPP_

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

#include "fp32_xmm.hpp"
#include "generic_vector.hpp"
#include "vector_load.hpp"
#include "vector_store.hpp"
#include "type_conversions.hpp"

namespace SIMD_NAMESPACE
{
#if COMPILED_WITH_AVX512F

	template<>
	class Vector<float, ZMM>
	{
		private:
			__m512 m_data;
			static inline __m512 broadcast(float x) noexcept
			{
				return _mm512_set1_ps(x);
			}
		public:
			Vector() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			template<typename T>
			Vector(const T *src, int num = size()) noexcept // @suppress("Class members should be properly initialized")
			{
				load(src, num);
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
					m_data(broadcast(Converter<SCALAR, float16, float>()(x)))
			{
			}
			Vector(__m512i raw_bytes) noexcept :
					m_data(_mm512_castsi512_ps(raw_bytes))
			{
			}
			Vector(__m512 x) noexcept :
					m_data(x)
			{
			}
			Vector<float, ZMM>& operator=(__m512 x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m512() const noexcept
			{
				return m_data;
			}
			__m512i raw_bytes() const noexcept
			{
				return _mm512_castps_si512(m_data);
			}
			void load(const float *src, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				m_data = Loader<ZMM>()(src, num);
			}
			void load(const float16 *src, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				const __m256i tmp = Loader<XMM>()(reinterpret_cast<const uint16_t*>(src), num);
				m_data = Converter<ZMM, float16, float>()(tmp);
			}
			void store(float *dst, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				Storer<ZMM>()(dst, m_data, num);
			}
			void store(float16 *dst, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				const __m256i tmp = Converter<ZMM, float, float16>()(m_data);
				Storer<XMM>()(reinterpret_cast<uint16_t*>(dst), tmp, num);
			}

			static Vector<float, ZMM> zero() noexcept
			{
				return Vector<float, ZMM>(0.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, ZMM> one() noexcept
			{
				return Vector<float, ZMM>(1.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, ZMM> epsilon() noexcept
			{
				return Vector<float, ZMM>(std::numeric_limits<float>::epsilon()); // @suppress("Ambiguous problem")
			}

			static constexpr int size() noexcept
			{
				return 16;
			}
			static constexpr RegisterType register_type() noexcept
			{
				return ZMM;
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
	 */
	static inline Vector<float, ZMM> operator==(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 0);
	}
	static inline Vector<float, ZMM> operator!=(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 4);
	}
	static inline Vector<float, ZMM> operator<(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 1);
	}
	static inline Vector<float, ZMM> operator<=(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 2);
	}
	static inline Vector<float, ZMM> operator>(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 14);
	}
	static inline Vector<float, ZMM> operator>=(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_cmp_ps(lhs, rhs, 13);
	}
	static inline Vector<float, ZMM> operator&(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_and_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> operator|(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_or_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> operator^(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_xor_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> operator~(Vector<float, ZMM> x) noexcept
	{
#if COMPILED_WITH_AVX2
		const __m256i y = _mm256_castps_si256(x);
		const __m256i tmp = _mm256_cmpeq_epi32(y, y);
#else
		const __m256i tmp = ymm_constant<0xFFFFFFFFu>();
#endif
		return _mm256_xor_ps(x, _mm256_castsi256_ps(tmp));
	}
	static inline Vector<float, ZMM> operator!(Vector<float, ZMM> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline Vector<float, ZMM> operator+(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_add_ps(lhs, rhs);
	}

	static inline Vector<float, ZMM> operator-(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_sub_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> operator-(Vector<float, ZMM> x) noexcept
	{
		return _mm256_xor_ps(x, Vector<float, ZMM>(-0.0f)); // @suppress("Ambiguous problem")
	}

	static inline Vector<float, ZMM> operator*(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_mul_ps(lhs, rhs);
	}

	static inline Vector<float, ZMM> operator/(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
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
	static inline Vector<float, ZMM> select(Vector<float, ZMM> mask, Vector<float, ZMM> x, Vector<float, ZMM> y)
	{
		return _mm256_blendv_ps(y, x, mask);
	}

	/* Float vector functions */
	static inline Vector<float, ZMM> max(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_max_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> min(Vector<float, ZMM> lhs, Vector<float, ZMM> rhs) noexcept
	{
		return _mm256_min_ps(lhs, rhs);
	}
	static inline Vector<float, ZMM> abs(Vector<float, ZMM> x) noexcept
	{
		return _mm256_and_ps(x, _mm256_castsi256_ps(ymm_constant<0x7FFFFFFFu>()));
	}
	static inline Vector<float, ZMM> sqrt(Vector<float, ZMM> x) noexcept
	{
		return _mm256_sqrt_ps(x);
	}
	static inline Vector<float, ZMM> rsqrt(Vector<float, ZMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm256_rsqrt_ps(x);
#else
		return Vector<float, ZMM>::one() / sqrt(x);
#endif
	}
	static inline Vector<float, ZMM> rcp(Vector<float, ZMM> x) noexcept
	{
#if ENABLE_FAST_MATH
		return _mm256_rcp_ps(x);
#else
		return Vector<float, ZMM>::one() / x;
#endif
	}
	static inline Vector<float, ZMM> sgn(Vector<float, ZMM> x) noexcept
	{
		__m256 zero = _mm256_setzero_ps();
		__m256 positive = _mm256_and_ps(_mm256_cmp_ps(zero, x, 1), _mm256_set1_ps(1.0f));
		__m256 negative = _mm256_and_ps(_mm256_cmp_ps(x, zero, 1), _mm256_set1_ps(-1.0f));
		return _mm256_or_ps(positive, negative);
	}
	static inline Vector<float, ZMM> floor(Vector<float, ZMM> x) noexcept
	{
		return _mm256_floor_ps(x);
	}
	static inline Vector<float, ZMM> ceil(Vector<float, ZMM> x) noexcept
	{
		return _mm256_ceil_ps(x);
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline Vector<float, ZMM> mul_add(Vector<float, ZMM> a, Vector<float, ZMM> b, Vector<float, ZMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fmadd_ps(a, b, c);
#else
		return a * b + c;
#endif
	}
	/* Calculates a * b - c */
	static inline Vector<float, ZMM> mul_sub(Vector<float, ZMM> a, Vector<float, ZMM> b, Vector<float, ZMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fmsub_ps(a, b, c);
#else
		return a * b - c;
#endif
	}
	/* Calculates - a * b + c */
	static inline Vector<float, ZMM> neg_mul_add(Vector<float, ZMM> a, Vector<float, ZMM> b, Vector<float, ZMM> c) noexcept
	{
#if COMPILED_WITH_FMA
		return _mm256_fnmadd_ps(a, b, c);
#else
		return -a * b + c;
#endif
	}
	/* Calculates - a * b - c */
	static inline Vector<float, ZMM> neg_mul_sub(Vector<float, ZMM> a, Vector<float, ZMM> b, Vector<float, ZMM> c) noexcept
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
	static inline float horizontal_add(Vector<float, ZMM> x) noexcept
	{
		const __m128 y = _mm_add_ps(x.low(), x.high());
		return horizontal_add(Vector<float, XMM>(y));
	}
	static inline float horizontal_mul(Vector<float, ZMM> x) noexcept
	{
		const __m128 y = _mm_mul_ps(x.low(), x.high());
		return horizontal_mul(Vector<float, XMM>(y));
	}
	static inline float horizontal_min(Vector<float, ZMM> x) noexcept
	{
		const __m128 y = _mm_min_ps(x.low(), x.high());
		return horizontal_min(Vector<float, XMM>(y));
	}
	static inline float horizontal_max(Vector<float, ZMM> x) noexcept
	{
		const __m128 y = _mm_max_ps(x.low(), x.high());
		return horizontal_max(Vector<float, XMM>(y));
	}

#endif /* SUPPORTS_AVX */

} /* namespace SIMD_NAMESPACE */

#endif /* BACKEND_CPU_VECTORS_FP32_ZMM_HPP_ */
