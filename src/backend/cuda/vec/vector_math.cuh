/*
 * vector_math.cuh
 *
 *  Created on: Nov 3, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VECTOR_MATH_CUH_
#define BACKEND_CUDA_VEC_VECTOR_MATH_CUH_

#include "utils.cuh"
#include "vec1d.cuh"
#include "vec1f.cuh"
#include "vec2f.cuh"
#include "vec4f.cuh"
#include "vec1h.cuh"
#include "vec2h.cuh"
#include "vec4h.cuh"
#include "vec8h.cuh"
#include "vector_load.cuh"
#include "vector_store.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace vectors
{
	template<int N, typename T, typename U>
	__device__ void vector_copy(T *dst, const U *src)
	{
		store_vec(dst, load_vec<U, N>(src));
	}

	template<typename T, int N>
	__device__ vec<T, N> zero()
	{
		return vec<T, N>(0.0f);
	}
	template<typename T, int N>
	__device__ vec<T, N> one()
	{
		return vec<T, N>(1.0f);
	}

	template<typename T, int N>
	__device__ vec<T, N> sigmoid(const vec<T, N> &x)
	{
		return one<T, N>() / (one<T, N>() + vectors::exp(-x));
	}
	template<typename T, int N>
	__device__ vec<T, N> relu(const vec<T, N> &x)
	{
		return vectors::max(zero<T, N>(), x);
	}
	template<typename T, int N>
	__device__ vec<T, N> approx_gelu(const vec<T, N> &x)
	{
		return x * vectors::sigmoid(vec<T, N>(1.6849f) * x);
	}

}

#endif /* BACKEND_CUDA_VEC_VECTOR_MATH_CUH_ */
