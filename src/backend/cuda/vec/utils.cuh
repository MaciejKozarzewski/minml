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

#define BF16_MIN_ARCH 800

#define FP16_MIN_ARCH 700

namespace vectors
{
	template<int N>
	DEVICE_INLINE bool is_aligned(const void *ptr)
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % N) == 0;
	}

	DEVICE_INLINE uint64_t as_uint(double x)
	{
		assert(sizeof(x) == sizeof(uint64_t));
		return reinterpret_cast<uint64_t*>(&x)[0];
	}
	DEVICE_INLINE uint32_t as_uint(float x)
	{
		assert(sizeof(x) == sizeof(uint32_t));
		return reinterpret_cast<uint32_t*>(&x)[0];
	}
	DEVICE_INLINE uint32_t as_uint(half2 x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		assert(sizeof(x) == sizeof(uint32_t));
		return reinterpret_cast<uint32_t*>(&x)[0];
#else
		return uint32_t{};
#endif
	}
	DEVICE_INLINE uint16_t as_uint(half x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		assert(sizeof(x) == sizeof(uint16_t));
		return reinterpret_cast<uint16_t*>(&x)[0];
#else
		return uint16_t{};
#endif
	}

	DEVICE_INLINE double as_double(uint64_t x)
	{
		assert(sizeof(x) == sizeof(double));
		return reinterpret_cast<double*>(&x)[0];
	}
	DEVICE_INLINE float as_float(uint32_t x)
	{
		assert(sizeof(x) == sizeof(float));
		return reinterpret_cast<float*>(&x)[0];
	}
	DEVICE_INLINE half2 as_half2(uint32_t x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		assert(sizeof(x) == sizeof(half2));
		return reinterpret_cast<half2*>(&x)[0];
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half as_half(uint16_t x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		assert(sizeof(x) == sizeof(half));
		return reinterpret_cast<half*>(&x)[0];
#else
		return half{};
#endif
	}

	DEVICE_INLINE double bit_invert(double x)
	{
		return as_double(~as_uint(x));
	}
	DEVICE_INLINE float bit_invert(float x)
	{
		return as_float(~as_uint(x));
	}
	DEVICE_INLINE half2 bit_invert(half2 x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half2(~as_uint(x));
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half bit_invert(half x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half(~as_uint(x));
#else
		return half{};
#endif
	}

	template<typename T>
	DEVICE_INLINE T to_mask(bool b);

	template<>
	DEVICE_INLINE half to_mask(bool b)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return b ? as_half(0xFFFFu) : as_half(0x0000u);
#else
		return half{};
#endif
	}
	template<>
	DEVICE_INLINE float to_mask(bool b)
	{
		return b ? as_float(0xFFFFFFFFu) : as_float(0x00000000u);
	}
	template<>
	DEVICE_INLINE double to_mask(bool b)
	{
		return b ? as_double(0xFFFFFFFFFFFFFFFFu) : as_double(0x0000000000000000u);
	}

	template<typename T>
	DEVICE_INLINE T logical_or(T lhs, T rhs);
	template<typename T>
	DEVICE_INLINE T logical_and(T lhs, T rhs);
	template<typename T>
	DEVICE_INLINE T logical_xor(T lhs, T rhs);

	template<>
	DEVICE_INLINE half logical_or(half lhs, half rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half(as_uint(lhs) | as_uint(rhs));
#else
		return half{};
#endif
	}
	template<>
	DEVICE_INLINE half logical_and(half lhs, half rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half(as_uint(lhs) & as_uint(rhs));
#else
		return half{};
#endif
	}
	template<>
	DEVICE_INLINE half logical_xor(half lhs, half rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half(as_uint(lhs) ^ as_uint(rhs));
#else
		return half{};
#endif
	}

	template<>
	DEVICE_INLINE half2 logical_or(half2 lhs, half2 rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half2(as_uint(lhs) | as_uint(rhs));
#else
		return half2{};
#endif
	}
	template<>
	DEVICE_INLINE half2 logical_and(half2 lhs, half2 rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half2(as_uint(lhs) & as_uint(rhs));
#else
		return half2{};
#endif
	}
	template<>
	DEVICE_INLINE half2 logical_xor(half2 lhs, half2 rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return as_half2(as_uint(lhs) ^ as_uint(rhs));
#else
		return half2{};
#endif
	}

	template<>
	DEVICE_INLINE float logical_or(float lhs, float rhs)
	{
		return as_float(as_uint(lhs) | as_uint(rhs));
	}
	template<>
	DEVICE_INLINE float logical_and(float lhs, float rhs)
	{
		return as_float(as_uint(lhs) & as_uint(rhs));
	}
	template<>
	DEVICE_INLINE float logical_xor(float lhs, float rhs)
	{
		return as_float(as_uint(lhs) ^ as_uint(rhs));
	}

	template<>
	DEVICE_INLINE double logical_or(double lhs, double rhs)
	{
		return as_double(as_uint(lhs) | as_uint(rhs));
	}
	template<>
	DEVICE_INLINE double logical_and(double lhs, double rhs)
	{
		return as_double(as_uint(lhs) & as_uint(rhs));
	}
	template<>
	DEVICE_INLINE double logical_xor(double lhs, double rhs)
	{
		return as_double(as_uint(lhs) ^ as_uint(rhs));
	}

	template<typename T>
	DEVICE_INLINE bool is_true(T b)
	{
		return as_uint(b) != 0;
	}
	template<typename T>
	DEVICE_INLINE bool is_false(T b)
	{
		return as_uint(b) == 0;
	}

	/*
	 * comparison operators
	 */
	DEVICE_INLINE half2 half2_to_mask(bool bx, bool by)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2(bx ? as_half(0xFFFFu) : as_half(0x0000u), by ? as_half(0xFFFFu) : as_half(0x0000u));
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_eq(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x == rhs.x, lhs.y == rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_neq(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x != rhs.x, lhs.y != rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_gt(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x > rhs.x, lhs.y > rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_ge(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x >= rhs.x, lhs.y >= rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_lt(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x < rhs.x, lhs.y < rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_compare_le(const half2 &lhs, const half2 &rhs)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2_to_mask(lhs.x <= rhs.x, lhs.y <= rhs.y);
#else
		return half2{};
#endif
	}
	DEVICE_INLINE half2 half2_select(const half2 &cond, const half2 &a, const half2 &b)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return half2(is_true(cond.x) ? a.x : b.x, is_true(cond.y) ? a.y : b.y);
#else
		return half2{};
#endif
	}

}
/* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_UTILS_CUH_ */
