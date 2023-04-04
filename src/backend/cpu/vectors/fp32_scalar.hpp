/*
 * fp32_scalar.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_SCALAR_HPP_
#define VECTORS_FP32_SCALAR_HPP_

#include <cassert>
#include <algorithm>
#include <cmath>

#include "generic_vector.hpp"
#include "vector_load.hpp"
#include "vector_store.hpp"
#include "vector_utils.hpp"
#include "types.hpp"
#include "type_conversions.hpp"

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
			Vector(const T *src, int num = size()) noexcept // @suppress("Class members should be properly initialized")
			{
				load(src, num);
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
					m_data(Converter<SCALAR, float16, float>()(x))
			{
			}
			Vector(sw_float16 x) noexcept :
					m_data(Converter<SCALAR, sw_float16, float>()(x))
			{
			}
			Vector(bfloat16 x) noexcept :
					m_data(Converter<SCALAR, bfloat16, float>()(x))
			{
			}
			Vector(sw_bfloat16 x) noexcept :
					m_data(Converter<SCALAR, sw_bfloat16, float>()(x))
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
			void load(const float *ptr, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				m_data = Loader<SCALAR>()(ptr, num);
			}
			void load(const float16 *ptr, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				const float16 tmp = Loader<SCALAR>()(ptr, num);
				m_data = Converter<SCALAR, float16, float>()(tmp);
			}
			void load(const sw_float16 *ptr, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				const sw_float16 tmp = Loader<SCALAR>()(ptr, num);
				m_data = Converter<SCALAR, sw_float16, float>()(tmp);
			}
			void load(const bfloat16 *ptr, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				const bfloat16 tmp = Loader<SCALAR>()(ptr, num);
				m_data = Converter<SCALAR, bfloat16, float>()(tmp);
			}
			void load(const sw_bfloat16 *ptr, int num = size()) noexcept
			{
				assert(0 <= num && num <= size());
				const sw_bfloat16 tmp = Loader<SCALAR>()(ptr, num);
				m_data = Converter<SCALAR, sw_bfloat16, float>()(tmp);
			}
			void store(float *ptr, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				Storer<SCALAR>()(ptr, m_data, num);
			}
			void store(float16 *ptr, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				const float16 tmp = Converter<SCALAR, float, float16>()(m_data);
				Storer<SCALAR>()(ptr, tmp, num);
			}
			void store(sw_float16 *ptr, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				const sw_float16 tmp = Converter<SCALAR, float, sw_float16>()(m_data);
				Storer<SCALAR>()(ptr, tmp, num);
			}
			void store(bfloat16 *ptr, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				const bfloat16 tmp = Converter<SCALAR, float, bfloat16>()(m_data);
				Storer<SCALAR>()(ptr, tmp, num);
			}
			void store(sw_bfloat16 *ptr, int num = size()) const noexcept
			{
				assert(0 <= num && num <= size());
				const sw_bfloat16 tmp = Converter<SCALAR, float, sw_bfloat16>()(m_data);
				Storer<SCALAR>()(ptr, tmp, num);
			}
			template<typename T>
			void insert(T value, int index) noexcept
			{
				assert(0 <= index && index < size());
				m_data = Converter<SCALAR, T, float>()(value);
			}
			float extract(int index) const noexcept
			{
				assert(0 <= index && index < size());
				return m_data;
			}
			float operator[](int index) const noexcept
			{
				return extract(index);
			}
			void cutoff(const int num, Vector<float, SCALAR> value = zero()) noexcept
			{
				assert(0 <= num && num <= size());
				m_data = (num == 0) ? value.m_data : m_data;
			}

			static constexpr float scalar_zero() noexcept
			{
				return 0.0f;
			}
			static constexpr float scalar_one() noexcept
			{
				return 1.0f;
			}
			static constexpr float scalar_epsilon() noexcept
			{
				return std::numeric_limits<float>::epsilon();
			}

			static Vector<float, SCALAR> zero() noexcept
			{
				return Vector<float, SCALAR>(scalar_zero()); // @suppress("Ambiguous problem")
			}
			static Vector<float, SCALAR> one() noexcept
			{
				return Vector<float, SCALAR>(scalar_one()); // @suppress("Ambiguous problem")
			}
			static Vector<float, SCALAR> epsilon() noexcept
			{
				return Vector<float, SCALAR>(scalar_epsilon()); // @suppress("Ambiguous problem")
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
