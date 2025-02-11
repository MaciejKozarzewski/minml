/*
 * convert.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_CONVERT_CUH_
#define BACKEND_CUDA_VEC_CONVERT_CUH_

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

	template<typename T, typename U, int N>
	DEVICE_INLINE vec<T, N> convert(const vec<U, N> &a);

	/*
	 * fp64 -> fp64
	 */
	template<>
	DEVICE_INLINE vec<double, 1> convert(const vec<double, 1> &a)
	{
		return a;
	}

	/*
	 * fp32 -> fp32
	 */
	template<>
	DEVICE_INLINE vec<float, 1> convert(const vec<float, 1> &a)
	{
		return a;
	}
	template<>
	DEVICE_INLINE vec<float, 2> convert(const vec<float, 2> &a)
	{
		return a;
	}
	template<>
	DEVICE_INLINE vec<float, 4> convert(const vec<float, 4> &a)
	{
		return a;
	}

	/*
	 * fp16 -> fp16
	 */
	template<>
	DEVICE_INLINE vec<half, 1> convert(const vec<half, 1> &a)
	{
		return a;
	}
	template<>
	DEVICE_INLINE vec<half, 2> convert(const vec<half, 2> &a)
	{
		return a;
	}
	template<>
	DEVICE_INLINE vec<half, 4> convert(const vec<half, 4> &a)
	{
		return a;
	}
	template<>
	DEVICE_INLINE vec<half, 8> convert(const vec<half, 8> &a)
	{
		return a;
	}

	/*
	 * fp16 -> fp32 vector conversion
	 */
	template<>
	DEVICE_INLINE vec<float, 1> convert(const vec<half, 1> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<float, 1>(__half2float(a.x0));
#else
		return vec<float, 1> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<float, 2> convert(const vec<half, 2> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<float, 2>(__half2float(a.x0.x), __half2float(a.x0.y));
#else
		return vec<float, 2> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<float, 4> convert(const vec<half, 4> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<float, 4>(__half2float(a.x0.x), __half2float(a.x0.y), __half2float(a.x1.x), __half2float(a.x1.y));
#else
		return vec<float, 4> { };
#endif
	}

	/*
	 * fp32 -> fp16 vector conversion
	 */
	template<>
	DEVICE_INLINE vec<half, 1> convert(const vec<float, 1> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<half, 1>(__float2half(a.x0));
#else
		return vec<half, 1> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<half, 2> convert(const vec<float, 2> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<half, 2>(__float2half(a.x0), __float2half(a.x1));
#else
		return vec<half, 2> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<half, 4> convert(const vec<float, 4> &a)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return vec<half, 4>(__float2half(a.x0), __float2half(a.x1), __float2half(a.x2), __float2half(a.x3));
#else
		return vec<half, 4> { };
#endif
	}

} /* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_CONVERT_CUH_ */
