/*
 * generic_vec.cuh
 *
 *  Created on: Jul 29, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_GENERIC_VEC_CUH_
#define BACKEND_CUDA_VEC_GENERIC_VEC_CUH_

#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace vectors2
{
	template<typename T, int N>
	struct vec
	{
			__device__ vec()
			{
			}
			__device__ vec(T x)
			{
			}
	};

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> square(const vec<T, N> &x)
	{
		return x * x;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> cube(const vec<T, N> &x)
	{
		return x * x * x;
	}

	/*
	 * arithmetic operations
	 */
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator+(const vec<T, N> &a)
	{
		return a;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator-(const vec<T, N> &a)
	{
		return -a;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator+(const vec<T, N> &lhs, T rhs)
	{
		return lhs + vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator+(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) + rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator+=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator+=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator-(const vec<T, N> &lhs, T rhs)
	{
		return lhs - vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator-(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) - rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator-=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator-=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator*(const vec<T, N> &lhs, T rhs)
	{
		return lhs * vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator*(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) * rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator*=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator*=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator/(const vec<T, N> &lhs, T rhs)
	{
		return lhs / vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator/(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) / rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator/=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N>& operator/=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator+(const vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator-(const vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator*(const vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator/(const vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		return vec<T, N> { };
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator|(const vec<T, N> &lhs, T rhs)
	{
		return lhs | vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator|(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) | rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator&(const vec<T, N> &lhs, T rhs)
	{
		return lhs | vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator&(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) | rhs;
	}

	/*
	 * comparison operators
	 */
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator>(const vec<T, N> &lhs, T rhs)
	{
		return lhs > vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator>(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) > rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator>=(const vec<T, N> &lhs, T rhs)
	{
		return lhs >= vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator>=(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) >= rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator<(const vec<T, N> &lhs, T rhs)
	{
		return rhs >= lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator<(T lhs, const vec<T, N> &rhs)
	{
		return rhs >= lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator<=(const vec<T, N> &lhs, T rhs)
	{
		return rhs > lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator<=(T lhs, const vec<T, N> &rhs)
	{
		return rhs > lhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator==(const vec<T, N> &lhs, T rhs)
	{
		return lhs == vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator==(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) == rhs;
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator!=(const vec<T, N> &lhs, T rhs)
	{
		return lhs != vec<T, N>(rhs);
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> operator!=(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) != rhs;
	}

	/*
	 * mathematical functions
	 */
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> abs(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> max(const vec<T, N> &a, const vec<T, N> &b)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> min(const vec<T, N> &a, const vec<T, N> &b)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> ceil(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> floor(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> sqrt(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> pow(const vec<T, N> &a, const vec<T, N> &b)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> mod(const vec<T, N> &a, const vec<T, N> &b)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> exp(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> log(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> tanh(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> expm1(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> log1p(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> sin(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> cos(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}
	template<typename T, int N>
	DEVICE_INLINE vec<T, N> tan(const vec<T, N> &a)
	{
		return vec<T, N> { };
	}

	template<typename T, int N>
	DEVICE_INLINE vec<T, N> select(const vec<T, N> &cond, const vec<T, N> &a, const vec<T, N> &b)
	{
		return vec<T, N> { };
	}

	/*
	 * reductions
	 */
	template<typename T, int N>
	DEVICE_INLINE T horizontal_add(const vec<T, N> &a)
	{
		return T { };
	}
	template<typename T, int N>
	DEVICE_INLINE T horizontal_max(const vec<T, N> &a)
	{
		return T { };
	}
	template<typename T, int N>
	DEVICE_INLINE T horizontal_min(const vec<T, N> &a)
	{
		return T { };
	}

	template<typename T, int N>
	DEVICE_INLINE void atomic_add(T *address, const vec<T, N> &value)
	{
	}

}
/* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_GENERIC_VEC_CUH_ */
