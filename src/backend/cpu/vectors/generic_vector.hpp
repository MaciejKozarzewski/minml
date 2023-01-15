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
#include "vector_macros.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T, class dummy = T>
	class Vector;

	/*
	 * Bitwise operations.
	 */
	template<typename T, typename U = T>
	static inline Vector<T>& operator&(Vector<T> lhs, U rhs) noexcept
	{
		return lhs & Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator&(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) & rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator&=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs & rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline Vector<T>& operator|(Vector<T> lhs, U rhs) noexcept
	{
		return lhs | Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator|(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) | rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator|=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs | rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline Vector<T>& operator^(Vector<T> lhs, U rhs) noexcept
	{
		return lhs ^ Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator^(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) ^ rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator^=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs ^ rhs);
		return lhs;
	}

	/*
	 * Bitwise shifts.
	 */
	template<typename T>
	static inline Vector<T>& operator>>=(Vector<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs >> rhs);
		return lhs;
	}
	template<typename T>
	static inline Vector<T>& operator<<=(Vector<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs << rhs);
		return lhs;
	}

	/*
	 * Compare operations.
	 */
	template<typename T, typename U = T>
	static inline Vector<T> operator<(Vector<T> lhs, U rhs) noexcept
	{
		return lhs < Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator<(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) < rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator<=(Vector<T> lhs, U rhs) noexcept
	{
		return lhs <= Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator<=(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) <= rhs;
	}

	template<typename T, typename U = T>
	static inline Vector<T> operator>(Vector<T> lhs, U rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator>(U lhs, Vector<T> rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator>=(Vector<T> lhs, U rhs) noexcept
	{
		return lhs < rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator>=(U lhs, Vector<T> rhs) noexcept
	{
		return lhs < rhs;
	}

	/*
	 * Arithmetic operations
	 */
	template<typename T, typename U = T>
	static inline Vector<T> operator+(Vector<T> lhs, U rhs) noexcept
	{
		return lhs + Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator+(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) + rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator+=(Vector<T> &lhs, Vector<U> rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator+=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T>
	static inline Vector<T> operator+(Vector<T> x) noexcept
	{
		return x;
	}

	template<typename T, typename U = T>
	static inline Vector<T> operator-(Vector<T> lhs, U rhs) noexcept
	{
		return lhs - Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator-(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) - rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator-=(Vector<T> &lhs, Vector<U> rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator-=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, typename U = T>
	static inline Vector<T> operator*(Vector<T> lhs, U rhs) noexcept
	{
		return lhs * Vector<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline Vector<T> operator*(U lhs, Vector<T> rhs) noexcept
	{
		return Vector<U>(lhs) * rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator*=(Vector<T> &lhs, Vector<U> rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator*=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, typename U>
	static inline Vector<T> operator/(Vector<U> lhs, T rhs) noexcept
	{
		return lhs / Vector<U>(rhs);
	}
	template<typename T, typename U>
	static inline Vector<T> operator/(T lhs, Vector<U> rhs) noexcept
	{
		return Vector<U>(lhs) / rhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator/=(Vector<T> &lhs, Vector<U> rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline Vector<T>& operator/=(Vector<T> &lhs, U rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace Vector_NAMESPACE */

#endif /* VECTORS_GENERIC_VECTOR_HPP_ */
