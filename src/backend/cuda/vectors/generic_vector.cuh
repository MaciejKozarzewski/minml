/*
 * generic_vector.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef GENERIC_VECTOR_CUH_
#define GENERIC_VECTOR_CUH_

#include <cstdint>

#define DEVICE_INLINE __device__ __forceinline__

#if __has_include(<cuda_bf16.h>)
#  define HAS_BF16_HEADER 1
#else
#  define HAS_BF16_HEADER 0
#endif

#define BF16_COMPUTE_MIN_ARCH 800

#define FP16_COMPUTE_MIN_ARCH 700
#define FP16_STORAGE_MIN_ARCH 530

namespace internal
{
	template<typename T>
	__device__ T sgn(T x) noexcept
	{
		return (static_cast<T>(0.0) < x) - (x < static_cast<T>(0.0));
	}
} /* namespace internal */

namespace vectors
{
	template<typename T, class dummy = T>
	class Vector;

	template<typename T>
	__device__ bool is_aligned(const void * ptr)
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % sizeof(T)) == 0;
	}

	template<typename T>
	__device__ constexpr int vector_length()
	{
		return 1;
	}

	template<typename T>
	__device__ Vector<T> vector_zero()
	{
		return Vector<T>();
	}
	template<typename T>
	__device__ Vector<T> vector_one()
	{
		return Vector<T>();
	}
	template<typename T>
	__device__ Vector<T> vector_epsilon()
	{
		return Vector<T>();
	}

	template<typename T>
	__device__ Vector<T> square(Vector<T> x)
	{
		return x * x;
	}
	template<typename T>
	__device__ Vector<T> sgn(Vector<T> x) noexcept
	{
		if (x > vector_zero<T>())
			return vector_one<T>();
		else
		{
			if (x < vector_zero<T>())
				return -vector_one<T>();
			else
				return vector_zero<T>();
		}
	}

	template<typename T>
	__device__ Vector<T> operator+(const Vector<T> &lhs, T rhs)
	{
		return lhs + Vector<T>(rhs);
	}
	template<typename T>
	__device__ Vector<T> operator+(T lhs, const Vector<T> &rhs)
	{
		return Vector<T>(lhs) + rhs;
	}
	template<typename T>
	__device__ Vector<T>& operator+=(Vector<T> &lhs, const Vector<T> &rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T>
	__device__ Vector<T>& operator+=(Vector<T> &lhs, T rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}

	template<typename T>
	__device__ Vector<T> operator-(const Vector<T> &lhs, T rhs)
	{
		return lhs - Vector<T>(rhs);
	}
	template<typename T>
	__device__ Vector<T> operator-(T lhs, const Vector<T> &rhs)
	{
		return Vector<T>(lhs) - rhs;
	}
	template<typename T>
	__device__ Vector<T>& operator-=(Vector<T> &lhs, const Vector<T> &rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T>
	__device__ Vector<T>& operator-=(Vector<T> &lhs, T rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T>
	__device__ Vector<T> operator*(const Vector<T> &lhs, T rhs)
	{
		return lhs * Vector<T>(rhs);
	}
	template<typename T>
	__device__ Vector<T> operator*(T lhs, const Vector<T> &rhs)
	{
		return Vector<T>(lhs) * rhs;
	}
	template<typename T>
	__device__ Vector<T>& operator*=(Vector<T> &lhs, const Vector<T> &rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T>
	__device__ Vector<T>& operator*=(Vector<T> &lhs, T rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T>
	__device__ Vector<T> operator/(const Vector<T> &lhs, T rhs)
	{
		return lhs / Vector<T>(rhs);
	}
	template<typename T>
	__device__ Vector<T> operator/(T lhs, const Vector<T> &rhs)
	{
		return Vector<T>(lhs) / rhs;
	}
	template<typename T>
	__device__ Vector<T>& operator/=(Vector<T> &lhs, const Vector<T> &rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T>
	__device__ Vector<T>& operator/=(Vector<T> &lhs, T rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace vectors */

#endif /* GENERIC_VECTOR_CUH_ */
