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

#if __CUDA_ARCH__ >= FP16_MIN_ARCH
	template<>
	class __builtin_align__(8) vec<half, 4>
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
			HOST_DEVICE vec(float f) :
					vec4h(static_cast<half>(f))
			{
			}
			HOST_DEVICE vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			HOST_DEVICE void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				*this = reinterpret_cast<const vec4h*>(ptr)[0];
			}
			HOST_DEVICE void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec4h*>(ptr)[0] = *this;
			}
			HOST_DEVICE vec4h& operator=(float x)
			{
				const half tmp = static_cast<half>(x);
				x0 = half2 { tmp, tmp };
				x1 = half2 { tmp, tmp };
				return *this;
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

	DEVICE_INLINE vec4h operator+(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(lhs.x0 + rhs.x0, lhs.x1 + rhs.x1);
	}
	DEVICE_INLINE vec4h operator-(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(lhs.x0 - rhs.x0, lhs.x1 - rhs.x1);
	}
	DEVICE_INLINE vec4h operator*(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(lhs.x0 * rhs.x0, lhs.x1 * rhs.x1);
	}
	DEVICE_INLINE vec4h operator/(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(lhs.x0 / rhs.x0, lhs.x1 / rhs.x1);
	}

	DEVICE_INLINE vec4h abs(vec4h a)
	{
		return vec4h(__habs2(a.x0), __habs2(a.x1));
	}
	DEVICE_INLINE vec4h max(vec4h a, vec4h b)
	{
		return vec4h(__hmax2(a.x0, b.x0), __hmax2(a.x1, b.x1));
	}
	DEVICE_INLINE vec4h min(vec4h a, vec4h b)
	{
		return vec4h(__hmin2(a.x0, b.x0), __hmin2(a.x1, b.x1));
	}
	DEVICE_INLINE vec4h ceil(vec4h a)
	{
		return vec4h(h2ceil(a.x0), h2ceil(a.x1));
	}
	DEVICE_INLINE vec4h floor(vec4h a)
	{
		return vec4h(h2floor(a.x0), h2floor(a.x1));
	}
	DEVICE_INLINE vec4h sqrt(vec4h a)
	{
		return vec4h(h2sqrt(a.x0), h2sqrt(a.x1));
	}
	DEVICE_INLINE vec4h exp(vec4h a)
	{
		return vec4h(h2exp(a.x0), h2exp(a.x1));
	}
	DEVICE_INLINE vec4h log(vec4h a)
	{
		return vec4h(h2log(a.x0), h2log(a.x1));
	}
	DEVICE_INLINE vec4h tanh(vec4h a)
	{
		const vec4h p = exp(a);
		const vec4h m = exp(-a);
		return (p - m) / (p + m);
	}
	DEVICE_INLINE vec4h sin(vec4h a)
	{
		return vec4h(h2sin(a.x0), h2sin(a.x1));
	}
	DEVICE_INLINE vec4h cos(vec4h a)
	{
		return vec4h(h2cos(a.x0), h2cos(a.x1));
	}
	DEVICE_INLINE vec4h erf(const vec4h &a)
	{
		return tanh(vec4h(0.797884561f) * a * (vec4h(1.0f) + vec4h(0.044715f) * square(a)));
	}

	DEVICE_INLINE half horizontal_add(vec4h a)
	{
		const half2 tmp = a.x0 + a.x1;
		return tmp.x + tmp.y;
	}
	DEVICE_INLINE half horizontal_max(vec4h a)
	{
		const half2 tmp = __hmax2(a.x0, a.x1);
		return __hmax(tmp.x, tmp.y);
	}
	DEVICE_INLINE half horizontal_min(vec4h a)
	{
		const half2 tmp = __hmin2(a.x0, a.x1);
		return __hmin(tmp.x, tmp.y);
	}

	DEVICE_INLINE vec4h select(const vec4h &cond, const vec4h &a, const vec4h &b)
	{
		return vec4h(is_true(cond.x0.x) ? a.x0.x : b.x0.x, is_true(cond.x0.y) ? a.x0.y : b.x0.y,
				is_true(cond.x1.x) ? a.x1.x : b.x1.x, is_true(cond.x1.y) ? a.x1.y : b.x1.y);
	}
#endif
} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC4H_CUH_ */
