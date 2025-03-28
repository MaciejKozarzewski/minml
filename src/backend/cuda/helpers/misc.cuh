/*
 * misc.cuh
 *
 *  Created on: Mar 20, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_HELPERS_MISC_CUH_
#define BACKEND_CUDA_HELPERS_MISC_CUH_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cassert>

__device__ __forceinline__ bool is_inside(int idx, int range)
{
	return 0 <= idx && idx < range;
}

__device__ __forceinline__ bool is_inside(int h, int w, int height, int width)
{
	return 0 <= h && h < height && 0 <= w && w < width;
}

template<typename T>
__device__ __forceinline__ T get(float x);

template<>
__device__ __forceinline__ float get(float x)
{
	return x;
}
template<>
__device__ __forceinline__ half get(float x)
{
	return half(x);
}
template<>
__device__ __forceinline__ half2 get(float x)
{
	return half2(x, x);
}

/*
 * dst = src * beta + alpha * value;
 */
template<typename T, typename U, typename V>
__device__ void update_element(T *dst, const U *src, float alpha, V value, float beta)
{
	assert(dst != nullptr);
	if (beta == 0.0f)
	{
		*dst = static_cast<T>(get<V>(alpha) * value);
	}
	else
	{
		assert(src != nullptr);
		*dst = static_cast<T>(static_cast<V>(*src) * get<V>(beta) + get<V>(alpha) * value);
	}
}
template<typename T, typename U>
__device__ void update_element(T *dst, float alpha, U value, float beta)
{
	update_element(dst, dst, alpha, value, beta);
}

namespace ml
{
	namespace internal
	{
		__device__ __forceinline__ float relu(float x)
		{
			return max(0.0f, x);
		}
		__device__ __forceinline__ half relu(half x)
		{
			return __hmax(half(0.0f), x);
		}
		__device__ __forceinline__ half2 relu(half2 x)
		{
			return __hmax2(half2(0.0f, 0.0f), x);
		}

		__device__ __forceinline__ float tanh(float x)
		{
			return std::tanh(x);
		}
		__device__ __forceinline__ half tanh(half x)
		{
			const half p = hexp(x);
			const half m = hexp(-x);
			return (p - m) / (p + m);
		}
		__device__ __forceinline__ half2 tanh(half2 x)
		{
			const half2 p = h2exp(x);
			const half2 m = h2exp(-x);
			return (p - m) / (p + m);
		}

		__device__ __forceinline__ float sigmoid(float x)
		{
			return 1.0f / (1.0f + exp(-x));
		}
		__device__ __forceinline__ half sigmoid(half x)
		{
			const half one(1.0f);
			return one / (one + hexp(-x));
		}
		__device__ __forceinline__ half2 sigmoid(half2 x)
		{
			const half2 one(1.0f, 1.0f);
			return one / (one + h2exp(-x));
		}
	}
}

#endif /* BACKEND_CUDA_HELPERS_MISC_CUH_ */
