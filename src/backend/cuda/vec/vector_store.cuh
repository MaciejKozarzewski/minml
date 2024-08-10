/*
 * vector_store.cuh
 *
 *  Created on: Aug 9, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VECTOR_STORE_CUH_
#define BACKEND_CUDA_VEC_VECTOR_STORE_CUH_

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
	DEVICE_INLINE void store_vec(T *ptr, const vec<U, N> &value)
	{
		assert(ptr != nullptr);
		assert(is_aligned<sizeof(vec<U, N>)>(ptr));
		reinterpret_cast<vec<T, N>*>(ptr)[0] = convert<T, U, N>(value);
	}

	/*
	 * partial loads (also can be unaligned)
	 */

	template<typename T, int N, typename U>
	void partial_store_vec(T *ptr, const vec<U, N> &value, int num)
	{
		partial_store_vec<T, N>(ptr, convert<T, U, N>(value), num);
	}

	template<typename T>
	DEVICE_INLINE void partial_store_vec(T *ptr, const vec<T, 1> &value, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 1);
		if (num == 1)
			ptr[0] = value.x0;
	}
	DEVICE_INLINE void partial_store_vec(float *ptr, const vec<float, 2> &value, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 2);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = value.x0;
				break;
			default:
				ptr[0] = value.x0;
				ptr[1] = value.x1;
				break;
		}
	}
	DEVICE_INLINE void partial_store_vec(half *ptr, const vec<half, 2> &value, int num)
	{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		assert(ptr != nullptr);
		assert(0 <= num && num <= 2);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = value.x0.x;
				break;
			default:
				ptr[0] = value.x0.x;
				ptr[1] = value.x0.y;
				break;
		}
#endif
	}
	DEVICE_INLINE void partial_store_vec(float *ptr, const vec<float, 4> &value, int num)
	{
		assert(ptr != nullptr);
		assert(0 <= num && num <= 4);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = value.x0;
				break;
			case 2:
				ptr[0] = value.x0;
				ptr[1] = value.x1;
				break;
			case 3:
				ptr[0] = value.x0;
				ptr[1] = value.x1;
				ptr[2] = value.x2;
				break;
			default:
				ptr[0] = value.x0;
				ptr[1] = value.x1;
				ptr[2] = value.x2;
				ptr[3] = value.x3;
				break;
		}
	}
	DEVICE_INLINE void partial_store_vec(half *ptr, const vec<half, 4> &value, int num)
	{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		assert(ptr != nullptr);
		assert(0 <= num && num <= 4);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = value.x0.x;
				break;
			case 2:
				ptr[0] = value.x0.x;
				ptr[1] = value.x0.y;
				break;
			case 3:
				ptr[0] = value.x0.x;
				ptr[1] = value.x0.y;
				ptr[2] = value.x1.x;
				break;
			default:
				ptr[0] = value.x0.x;
				ptr[1] = value.x0.y;
				ptr[2] = value.x1.x;
				ptr[3] = value.x1.y;
				break;
		}
#endif
	}
}

#endif /* BACKEND_CUDA_VEC_VECTOR_STORE_CUH_ */
