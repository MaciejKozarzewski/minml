/*
 * load_store.cuh
 *
 *  Created on: Aug 9, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VECTOR_LOAD_CUH_
#define BACKEND_CUDA_VEC_VECTOR_LOAD_CUH_

#include "convert.cuh"
#include "vec4f.cuh"
#include "vec2f.cuh"
#include "vec1f.cuh"
#include "vec8h.cuh"
#include "vec4h.cuh"
#include "vec2h.cuh"
#include "vec1h.cuh"

#include <cassert>

namespace vectors2
{

	/*
	 * aligned full loads
	 */
	template<typename T, int N, typename U>
	DEVICE_INLINE vec<T, N> load_vec(const U *ptr)
	{
		assert(ptr != nullptr);
		return convert<T, U, N>(reinterpret_cast<const vec<U, N>*>(ptr)[0]);
	}

	/*
	 * partial loads (also can be unaligned)
	 */

	template<typename T, int N, typename U>
	DEVICE_INLINE vec<T, N> partial_load_vec(const U *ptr, int num)
	{
		return convert<T, U, N>(partial_load_vec<U, N>(ptr, num));
	}

	template<>
	DEVICE_INLINE vec<float, 1> partial_load_vec(const float *ptr, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 1);
		if (num == 0)
			return vec<float, 1>(0.0f);
		else
			return vec<float, 1>(ptr[0]);
	}
	template<>
	DEVICE_INLINE vec<float, 2> partial_load_vec(const float *ptr, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 2);
		switch (num)
		{
			case 0:
				return vec<float, 2>(0.0f, 0.0f);
			case 1:
				return vec<float, 2>(ptr[0], 0.0f);
			default:
				return vec<float, 2>(ptr[0], ptr[1]);
		}
	}
	template<>
	DEVICE_INLINE vec<float, 4> partial_load_vec(const float *ptr, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 4);
		switch (num)
		{
			case 0:
				return vec<float, 4>(0.0f);
			case 1:
				return vec<float, 4>(ptr[0], 0.0f, 0.0f, 0.0f);
			case 2:
				return vec<float, 4>(ptr[0], ptr[1], 0.0f, 0.0f);
			case 3:
				return vec<float, 4>(ptr[0], ptr[1], ptr[2], 0.0f);
			default:
				return vec<float, 4>(ptr[0], ptr[1], ptr[2], ptr[3]);
		}
	}

	template<>
	DEVICE_INLINE vec<half, 1> partial_load_vec(const half *ptr, int num)
	{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		assert(ptr != nullptr);
		assert(0 <= num && num <= 1);
		if (num == 0)
			return vec<half, 1>(0.0f);
		else
			return vec<half, 1>(ptr[0]);
#else
		return vec<half, 1> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<half, 2> partial_load_vec(const half *ptr, int num)
	{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		assert(ptr != nullptr);
		assert(0 <= num && num <= 2);
		switch (num)
		{
			case 0:
				return vec<half, 2>(0.0f, 0.0f);
			case 1:
				return vec<half, 2>(ptr[0], 0.0f);
			default:
				return vec<half, 2>(ptr[0], ptr[1]);
		}
#else
		return vec<half, 2> { };
#endif
	}
	template<>
	DEVICE_INLINE vec<half, 4> partial_load_vec(const half *ptr, int num)
	{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		assert(ptr != nullptr);
		assert(0 <= num && num <= 4);
		switch (num)
		{
			case 0:
				return vec<half, 4>(0.0f);
			case 1:
				return vec<half, 4>(ptr[0], 0.0f, 0.0f, 0.0f);
			case 2:
				return vec<half, 4>(ptr[0], ptr[1], 0.0f, 0.0f);
			case 3:
				return vec<half, 4>(ptr[0], ptr[1], ptr[2], 0.0f);
			default:
				return vec<half, 4>(ptr[0], ptr[1], ptr[2], ptr[3]);
		}
#else
		return vec<half, 4> { };
#endif
	}
}

#endif /* BACKEND_CUDA_VEC_VECTOR_LOAD_CUH_ */
