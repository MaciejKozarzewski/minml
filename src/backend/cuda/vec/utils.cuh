/*
 * utils.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_UTILS_CUH_
#define BACKEND_CUDA_VEC_UTILS_CUH_

#include <cuda_fp16.h>

#include <cstdint>
#include <cassert>

#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define HOST_DEVICE __host__ __device__

#define BF16_COMPUTE_MIN_ARCH 800

#define FP16_COMPUTE_MIN_ARCH 700
#define FP16_STORAGE_MIN_ARCH 530

namespace vectors2
{
	template<typename T>
	HOST_DEVICE bool is_aligned(const void *ptr)
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % sizeof(T)) == 0;
	}

	HOST_DEVICE_INLINE uint32_t as_uint(float x)
	{
		assert(sizeof(x) == sizeof(uint32_t));
		return reinterpret_cast<uint32_t*>(&x)[0];
	}
	HOST_DEVICE_INLINE uint32_t as_uint(half2 x)
	{
		assert(sizeof(x) == sizeof(uint32_t));
		return reinterpret_cast<uint32_t*>(&x)[0];
	}
	HOST_DEVICE_INLINE float as_float(uint32_t x)
	{
		assert(sizeof(x) == sizeof(float));
		return reinterpret_cast<float*>(&x)[0];
	}
	HOST_DEVICE_INLINE half2 as_half2(uint32_t x)
	{
		assert(sizeof(x) == sizeof(half2));
		return reinterpret_cast<half2*>(&x)[0];
	}

	HOST_DEVICE_INLINE float bit_invert(float x)
	{
		return as_float(~as_uint(x));
	}
	HOST_DEVICE_INLINE half2 bit_invert(half2 x)
	{
		return as_half2(~as_uint(x));
	}

} /* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_UTILS_CUH_ */
