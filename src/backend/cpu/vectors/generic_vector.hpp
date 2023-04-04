/*
 * generic_vector.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_GENERIC_VECTOR_HPP_
#define VECTORS_GENERIC_VECTOR_HPP_

#include <cstring>
#include <cmath>
#include <cinttypes>
#include <cassert>
#include <x86intrin.h>

#include "types.hpp"
#include "register_type.hpp"
#include "vector_macros.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T, RegisterType RT = AUTO>
	class Vector;

	/*
	 * Bitwise operations.
	 */
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator&(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs & Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator&(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) & rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator&=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = (lhs & rhs);
		return lhs;
	}

	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator|(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs | Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator|(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) | rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator|=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = (lhs | rhs);
		return lhs;
	}

	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator^(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs ^ Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator^(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) ^ rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator^=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = (lhs ^ rhs);
		return lhs;
	}

	/*
	 * Bitwise shifts.
	 */
	template<typename T, RegisterType RT>
	static inline Vector<T, RT>& operator>>=(Vector<T, RT> &lhs, T rhs) noexcept
	{
		lhs = (lhs >> rhs);
		return lhs;
	}
	template<typename T, RegisterType RT>
	static inline Vector<T, RT>& operator<<=(Vector<T, RT> &lhs, T rhs) noexcept
	{
		lhs = (lhs << rhs);
		return lhs;
	}

	/*
	 * Compare operations.
	 */
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator<(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs < (Vector<T, RT>(rhs));
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator<(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) < rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator<=(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs <= Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator<=(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) <= rhs;
	}

	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator>(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator>(U lhs, Vector<T, RT> rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator>=(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs < rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator>=(U lhs, Vector<T, RT> rhs) noexcept
	{
		return lhs < rhs;
	}

	/*
	 * Arithmetic operations
	 */
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator+(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs + Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator+(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) + rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator+=(Vector<T, RT> &lhs, Vector<U, RT> rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator+=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, RegisterType RT>
	static inline Vector<T, RT> operator+(Vector<T, RT> x) noexcept
	{
		return x;
	}

	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator-(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs - Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator-(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) - rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator-=(Vector<T, RT> &lhs, Vector<U, RT> rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator-=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator*(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs * Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT> operator*(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) * rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator*=(Vector<T, RT> &lhs, Vector<U, RT> rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator*=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, RegisterType RT, typename U>
	static inline Vector<T, RT> operator/(Vector<T, RT> lhs, U rhs) noexcept
	{
		return lhs / Vector<T, RT>(rhs);
	}
	template<typename T, RegisterType RT, typename U>
	static inline Vector<T, RT> operator/(U lhs, Vector<T, RT> rhs) noexcept
	{
		return Vector<T, RT>(lhs) / rhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator/=(Vector<T, RT> &lhs, Vector<U, RT> rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, RegisterType RT, typename U = T>
	static inline Vector<T, RT>& operator/=(Vector<T, RT> &lhs, U rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace Vector_NAMESPACE */

#endif /* VECTORS_GENERIC_VECTOR_HPP_ */
