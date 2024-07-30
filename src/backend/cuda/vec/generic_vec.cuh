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
	struct vec;

	template<typename T, int N>
	HOST_DEVICE vec<T, N> square(const vec<T, N> &x)
	{
		return x * x;
	}

	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator+(const vec<T, N> &lhs, T rhs)
	{
		return lhs + vec<T, N>(rhs);
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator+(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) + rhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator+=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator+=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}

	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator-(const vec<T, N> &lhs, T rhs)
	{
		return lhs - vec<T, N>(rhs);
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator-(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) - rhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator-=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator-=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator*(const vec<T, N> &lhs, T rhs)
	{
		return lhs * vec<T, N>(rhs);
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator*(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) * rhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator*=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator*=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator/(const vec<T, N> &lhs, T rhs)
	{
		return lhs / vec<T, N>(rhs);
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N> operator/(T lhs, const vec<T, N> &rhs)
	{
		return vec<T, N>(lhs) / rhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator/=(vec<T, N> &lhs, const vec<T, N> &rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, int N>
	HOST_DEVICE vec<T, N>& operator/=(vec<T, N> &lhs, T rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}

}
/* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_GENERIC_VEC_CUH_ */
