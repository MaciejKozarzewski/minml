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

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace vectors2
{

	template<typename T, int N>
	__device__ vec<T, N> sigmoid(const vec<T, N> &x)
	{
		const vec<T, N> one(static_cast<T>(1));
		return one / (one + vectors2::exp(-x));
	}
	template<typename T, int N>
	__device__ vec<T, N> relu(const vec<T, N> &x)
	{
		const vec<T, N> zero(static_cast<T>(0));
		return vectors2::max(zero, x);
	}
	template<typename T, int N>
	__device__ vec<T, N> approx_gelu(const vec<T, N> &x)
	{
		const vec<T, N> tmp(static_cast<T>(1.6849));
		return x * sigmoid(tmp * x);
	}

}

#endif /* BACKEND_CUDA_VEC_VECTOR_MATH_CUH_ */
