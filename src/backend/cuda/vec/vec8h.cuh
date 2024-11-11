/*
 * vec8h.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC8H_CUH_
#define BACKEND_CUDA_VEC_VEC8H_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors2
{
	using vec8h = vec<half, 8>;

#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
	template<>
	class __builtin_align__(16) vec<half, 8>
	{
		public:
			half2 x0, x1, x2, x3;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			HOST_DEVICE vec(float f) :
					vec8h(static_cast<half>(f))
			{
			}
			HOST_DEVICE vec(half2 h0, half2 h1, half2 h2, half2 h3) :
					x0(h0),
					x1(h1),
					x2(h2),
					x3(h3)
			{
			}
			HOST_DEVICE vec(half h) :
					vec8h(half2(h, h))
			{
			}
			HOST_DEVICE vec(half2 h) :
					vec8h(h, h, h, h)
			{
			}
			HOST_DEVICE vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			HOST_DEVICE void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				*this = reinterpret_cast<const vec8h*>(ptr)[0];
			}
			HOST_DEVICE void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec8h*>(ptr)[0] = *this;
			}
			HOST_DEVICE vec8h operator-() const
			{
				return vec8h(-x0, -x1, -x2, -x3);
			}
			HOST_DEVICE vec8h operator~() const
			{
				return vec8h(bit_invert(x0), bit_invert(x1), bit_invert(x2), bit_invert(x3));
			}
	};

	DEVICE_INLINE vec8h operator+(const vec8h &lhs, const vec8h &rhs)
	{
		return vec8h(lhs.x0 + rhs.x0, lhs.x1 + rhs.x1, lhs.x2 + rhs.x2, lhs.x3 + rhs.x3);
	}
	DEVICE_INLINE vec8h operator-(const vec8h &lhs, const vec8h &rhs)
	{
		return vec8h(lhs.x0 - rhs.x0, lhs.x1 - rhs.x1, lhs.x2 - rhs.x2, lhs.x3 - rhs.x3);
	}
	DEVICE_INLINE vec8h operator*(const vec8h &lhs, const vec8h &rhs)
	{
		return vec8h(lhs.x0 * rhs.x0, lhs.x1 * rhs.x1, lhs.x2 * rhs.x2, lhs.x3 * rhs.x3);
	}
	DEVICE_INLINE vec8h operator/(const vec8h &lhs, const vec8h &rhs)
	{
		return vec8h(lhs.x0 / rhs.x0, lhs.x1 / rhs.x1, lhs.x2 / rhs.x2, lhs.x3 / rhs.x3);
	}

	DEVICE_INLINE vec8h abs(vec8h a)
	{
		return vec8h(__habs2(a.x0), __habs2(a.x1), __habs2(a.x2), __habs2(a.x3));
	}
	DEVICE_INLINE vec8h max(vec8h a, vec8h b)
	{
		return vec8h(__hmax2(a.x0, b.x0), __hmax2(a.x1, b.x1), __hmax2(a.x2, b.x2), __hmax2(a.x3, b.x3));
	}
	DEVICE_INLINE vec8h min(vec8h a, vec8h b)
	{
		return vec8h(__hmin2(a.x0, b.x0), __hmin2(a.x1, b.x1), __hmin2(a.x2, b.x2), __hmin2(a.x3, b.x3));
	}
	DEVICE_INLINE vec8h ceil(vec8h a)
	{
		return vec8h(h2ceil(a.x0), h2ceil(a.x1), h2ceil(a.x2), h2ceil(a.x3));
	}
	DEVICE_INLINE vec8h floor(vec8h a)
	{
		return vec8h(h2floor(a.x0), h2floor(a.x1), h2floor(a.x2), h2floor(a.x3));
	}
	DEVICE_INLINE vec8h sqrt(vec8h a)
	{
		return vec8h(h2sqrt(a.x0), h2sqrt(a.x1), h2sqrt(a.x2), h2sqrt(a.x3));
	}
	DEVICE_INLINE vec8h exp(vec8h a)
	{
		return vec8h(h2exp(a.x0), h2exp(a.x1), h2exp(a.x2), h2exp(a.x3));
	}
	DEVICE_INLINE vec8h log(vec8h a)
	{
		return vec8h(h2log(a.x0), h2log(a.x1), h2log(a.x2), h2log(a.x3));
	}
	DEVICE_INLINE vec8h tanh(vec8h a)
	{
		const vec8h p = exp(a);
		const vec8h m = exp(-a);
		return (p - m) / (p + m);
	}
	DEVICE_INLINE vec8h sin(vec8h a)
	{
		return vec8h(h2sin(a.x0), h2sin(a.x1), h2sin(a.x2), h2sin(a.x3));
	}
	DEVICE_INLINE vec8h cos(vec8h a)
	{
		return vec8h(h2cos(a.x0), h2cos(a.x1), h2cos(a.x2), h2cos(a.x3));
	}
	DEVICE_INLINE vec8h erf(const vec8h &a)
	{
		return tanh(vec8h(0.797884561f) * a * (vec8h(1.0f) + vec8h(0.044715f) * square(a)));
	}

	DEVICE_INLINE half horizontal_add(vec8h a)
	{
		const half2 tmp = a.x0 + a.x1 + a.x2 + a.x3;
		return tmp.x + tmp.y;
	}
	DEVICE_INLINE half horizontal_max(vec8h a)
	{
		const half2 tmp = __hmax2(__hmax2(a.x0, a.x1), __hmax2(a.x2, a.x3));
		return __hmax(tmp.x, tmp.y);
	}
	DEVICE_INLINE half horizontal_min(vec8h a)
	{
		const half2 tmp = __hmin2(__hmin2(a.x0, a.x1), __hmin2(a.x2, a.x3));
		return __hmin(tmp.x, tmp.y);
	}

	DEVICE_INLINE vec8h select(const vec8h &cond, const vec8h &a, const vec8h &b)
	{
		return vec8h(half2{is_true(cond.x0.x) ? a.x0.x : b.x0.x, is_true(cond.x0.y) ? a.x0.y : b.x0.y},
					half2{is_true(cond.x1.x) ? a.x1.x : b.x1.x, is_true(cond.x1.y) ? a.x1.y : b.x1.y},
					half2{is_true(cond.x2.x) ? a.x2.x : b.x2.x, is_true(cond.x2.y) ? a.x2.y : b.x2.y},
					half2{is_true(cond.x3.x) ? a.x3.x : b.x3.x, is_true(cond.x3.y) ? a.x3.y : b.x3.y});
	}
#endif
} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC8H_CUH_ */
