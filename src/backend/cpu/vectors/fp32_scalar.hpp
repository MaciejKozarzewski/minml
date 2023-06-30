/*
 * fp32_scalar.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_SCALAR_HPP_
#define VECTORS_FP32_SCALAR_HPP_

#include "generic_vector.hpp"
#include "vector_utils.hpp"
#include "types.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>

namespace SIMD_NAMESPACE
{
	template<>
	class Vector<float, SCALAR>
	{
		private:
			float m_data;
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
					m_data(static_cast<float>(x))
			{
			}
			Vector(float x) noexcept :
					m_data(x)
			{
			}
			Vector(float16 x) noexcept :
#if COMPILED_WITH_F16C
					m_data(_cvtsh_ss(x.m_data))
#else
					m_data(0.0f)
#endif
			{
			}
			Vector(uint32_t raw_bytes) noexcept :
					m_data(bitwise_cast<float>(raw_bytes))
			{
			}
			operator float() const noexcept
			{
				return m_data;
			}
			uint32_t raw_bytes() const noexcept
			{
				return bitwise_cast<uint32_t>(m_data);
			}
			void load(const float *ptr) noexcept
			{
				assert(ptr != nullptr);
				m_data = ptr[0];
			}
			void partial_load(const float *ptr, int num) noexcept
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= size());
				m_data = (num == 1) ? ptr[0] : 0.0f;
			}
			void load(const float16 *ptr) noexcept
			{
				assert(ptr != nullptr);
#if COMPILED_WITH_F16C
				m_data = _cvtsh_ss(ptr[0].m_data);
#endif
			}
			void partial_load(const float16 *ptr, int num) noexcept
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= size());
				const float16 tmp = (num == 1) ? ptr[0] : float16 { 0u };
				load(&tmp);
			}
			void store(float *ptr) const noexcept
			{
				assert(ptr != nullptr);
				ptr[0] = m_data;
			}
			void partial_store(float *ptr, int num) const noexcept
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= size());
				if (num == 1)
					ptr[0] = m_data;
			}
			void store(float16 *ptr) const noexcept
			{
#if COMPILED_WITH_F16C
				assert(ptr != nullptr);
				ptr[0] = float16 { _cvtss_sh(m_data, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)) };
#endif
			}
			void partial_store(float16 *ptr, int num) const noexcept
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= size());
				float16 tmp;
				store(&tmp);
				if (num == 1)
					ptr[0] = tmp;
			}

			static Vector<float, SCALAR> zero() noexcept
			{
				return Vector<float, SCALAR>(0.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, SCALAR> one() noexcept
			{
				return Vector<float, SCALAR>(1.0f); // @suppress("Ambiguous problem")
			}
			static Vector<float, SCALAR> epsilon() noexcept
			{
				return Vector<float, SCALAR>(std::numeric_limits<float>::epsilon()); // @suppress("Ambiguous problem")
			}

			static constexpr int size() noexcept
			{
				return 1;
			}
			static constexpr RegisterType register_type() noexcept
			{
				return SCALAR;
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
	 */
	static inline Vector<float, SCALAR> operator==(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) == static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator!=(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) != static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator<(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) < static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator<=(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) <= static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator>(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) > static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator>=(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) >= static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u;
	}
	static inline Vector<float, SCALAR> operator&(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return lhs.raw_bytes() & rhs.raw_bytes();
	}
	static inline Vector<float, SCALAR> operator|(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return lhs.raw_bytes() | rhs.raw_bytes();
	}
	static inline Vector<float, SCALAR> operator^(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return lhs.raw_bytes() ^ rhs.raw_bytes();
	}
	static inline Vector<float, SCALAR> operator~(Vector<float, SCALAR> x) noexcept
	{
		return ~x.raw_bytes();
	}
	static inline Vector<float, SCALAR> operator!(Vector<float, SCALAR> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline Vector<float, SCALAR> operator+(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) + static_cast<float>(rhs);
	}

	static inline Vector<float, SCALAR> operator-(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) - static_cast<float>(rhs);
	}
	static inline Vector<float, SCALAR> operator-(Vector<float, SCALAR> x) noexcept
	{
		return -static_cast<float>(x);
	}

	static inline Vector<float, SCALAR> operator*(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) * static_cast<float>(rhs);
	}

	static inline Vector<float, SCALAR> operator/(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return static_cast<float>(lhs) / static_cast<float>(rhs);
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline Vector<float, SCALAR> select(Vector<float, SCALAR> mask, Vector<float, SCALAR> x, Vector<float, SCALAR> y)
	{
		return (mask.raw_bytes() == 0xFFFFFFFFu) ? x : y;
	}

	/* Float vector functions */
	static inline Vector<float, SCALAR> max(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return std::max(static_cast<float>(lhs), static_cast<float>(rhs));
	}
	static inline Vector<float, SCALAR> min(Vector<float, SCALAR> lhs, Vector<float, SCALAR> rhs) noexcept
	{
		return std::min(static_cast<float>(lhs), static_cast<float>(rhs));
	}
	static inline Vector<float, SCALAR> abs(Vector<float, SCALAR> x) noexcept
	{
		return std::fabs(static_cast<float>(x));
	}
	static inline Vector<float, SCALAR> sqrt(Vector<float, SCALAR> x) noexcept
	{
		return std::sqrt(static_cast<float>(x));
	}
	static inline Vector<float, SCALAR> rsqrt(Vector<float, SCALAR> x) noexcept
	{
		return 1.0f / std::sqrt(static_cast<float>(x));
	}
	static inline Vector<float, SCALAR> rcp(Vector<float, SCALAR> x) noexcept
	{
		return 1.0f / x;
	}
	static inline Vector<float, SCALAR> sgn(Vector<float, SCALAR> x) noexcept
	{
		return static_cast<float>((static_cast<float>(x) > 0.0f) - (static_cast<float>(x) < 0.0f));
	}
	static inline Vector<float, SCALAR> floor(Vector<float, SCALAR> x) noexcept
	{
		return std::floor(static_cast<float>(x));
	}
	static inline Vector<float, SCALAR> ceil(Vector<float, SCALAR> x) noexcept
	{
		return std::ceil(static_cast<float>(x));
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline Vector<float, SCALAR> mul_add(Vector<float, SCALAR> a, Vector<float, SCALAR> b, Vector<float, SCALAR> c) noexcept
	{
		return a * b + c;
	}
	/* Calculates a * b - c */
	static inline Vector<float, SCALAR> mul_sub(Vector<float, SCALAR> a, Vector<float, SCALAR> b, Vector<float, SCALAR> c) noexcept
	{
		return a * b - c;
	}
	/* Calculates - a * b + c */
	static inline Vector<float, SCALAR> neg_mul_add(Vector<float, SCALAR> a, Vector<float, SCALAR> b, Vector<float, SCALAR> c) noexcept
	{
		return -a * b + c;
	}
	/* Calculates - a * b - c */
	static inline Vector<float, SCALAR> neg_mul_sub(Vector<float, SCALAR> a, Vector<float, SCALAR> b, Vector<float, SCALAR> c) noexcept
	{
		return -a * b - c;
	}

	/*
	 * Horizontal functions
	 */

	static inline float horizontal_add(Vector<float, SCALAR> x) noexcept
	{
		return static_cast<float>(x);
	}
	static inline float horizontal_mul(Vector<float, SCALAR> x) noexcept
	{
		return static_cast<float>(x);
	}
	static inline float horizontal_min(Vector<float, SCALAR> x) noexcept
	{
		return static_cast<float>(x);
	}
	static inline float horizontal_max(Vector<float, SCALAR> x) noexcept
	{
		return static_cast<float>(x);
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP32_SCALAR_HPP_ */
