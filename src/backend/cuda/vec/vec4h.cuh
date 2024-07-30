/*
 * vec4h.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC4H_CUH_
#define BACKEND_CUDA_VEC_VEC4H_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors2
{
	using vec4h = vec<half, 4>;

	template<>
	class __builtin_align__(16) vec<half, 4>
	{
		public:
			half2 x0, x1;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			HOST_DEVICE vec(half2 h0, half2 h1) :
					x0(h0),
					x1(h1)
			{
			}
			HOST_DEVICE vec(half h0, half h1, half h2, half h3) :
					x0(h0, h1),
					x1(h2, h3)
			{
			}
			HOST_DEVICE vec(half2 h) :
					vec4h(h, h)
			{
			}
			HOST_DEVICE vec(half h) :
					vec4h(h, h, h, h)
			{
			}
			HOST_DEVICE vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			HOST_DEVICE void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				assert(is_aligned<vec4h>(ptr));
				*this = reinterpret_cast<const vec4h*>(ptr)[0];
			}
			HOST_DEVICE void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				assert(is_aligned<vec4h>(ptr));
				reinterpret_cast<vec4h*>(ptr)[0] = *this;
			}
			HOST_DEVICE vec4h operator-() const
			{
				return vec4h(-x0, -x1);
			}
			HOST_DEVICE vec4h operator~() const
			{
				return vec4h(bit_invert(x0), bit_invert(x1));
			}
	};

} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC4H_CUH_ */
