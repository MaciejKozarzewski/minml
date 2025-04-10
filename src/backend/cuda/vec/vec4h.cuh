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

namespace vectors
{
	using vec4h = vec<half, 4>;

	template<>
	class __builtin_align__(8) vec<half, 4>
	{
		public:
			half2 x0, x1;

			__device__ vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit __device__ vec(half2 h0, half2 h1) :
					x0(h0),
					x1(h1)
			{
			}
			explicit __device__ vec(half h0, half h1, half h2, half h3) :
					x0(h0, h1),
					x1(h2, h3)
			{
			}
			explicit __device__ vec(half2 h) :
					vec4h(h, h)
			{
			}
			explicit __device__ vec(half h) :
					vec4h(h, h, h, h)
			{
			}
			explicit __device__ vec(float f) :
					vec4h(static_cast<half>(f))
			{
			}
			__device__ vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			__device__ void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				*this = reinterpret_cast<const vec4h*>(ptr)[0];
			}
			__device__ void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec4h*>(ptr)[0] = *this;
			}
			__device__ vec4h& operator=(float x)
			{
				const half tmp = static_cast<half>(x);
				x0 = half2 { tmp, tmp };
				x1 = half2 { tmp, tmp };
				return *this;
			}
			__device__ vec4h operator-() const
			{
				return vec4h(-x0, -x1);
			}
			__device__ vec4h operator~() const
			{
				return vec4h(bit_invert(x0), bit_invert(x1));
			}
			__device__ int size() const
			{
				return 4;
			}
			__device__ half operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				switch (idx)
				{
					default:
					case 0:
						return x0.x;
					case 1:
						return x0.y;
					case 2:
						return x1.x;
					case 3:
						return x1.y;
				}
			}
			__device__ half& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				switch (idx)
				{
					default:
					case 0:
						return x0.x;
					case 1:
						return x0.y;
					case 2:
						return x1.x;
					case 3:
						return x1.y;
				}
			}
	};

#if __CUDA_ARCH__ >= FP16_MIN_ARCH
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec4h operator==(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_eq(lhs.x0, rhs.x0), half2_compare_eq(lhs.x1, rhs.x1));
	}
	DEVICE_INLINE vec4h operator!=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_neq(lhs.x0, rhs.x0), half2_compare_neq(lhs.x1, rhs.x1));
	}
	DEVICE_INLINE vec4h operator>(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_gt(lhs.x0, rhs.x0), half2_compare_gt(lhs.x1, rhs.x1));
	}
	DEVICE_INLINE vec4h operator>=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_ge(lhs.x0, rhs.x0), half2_compare_ge(lhs.x1, rhs.x1));
	}
	DEVICE_INLINE vec4h operator<(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_lt(lhs.x0, rhs.x0), half2_compare_lt(lhs.x1, rhs.x1));
	}
	DEVICE_INLINE vec4h operator<=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h(half2_compare_le(lhs.x0, rhs.x0), half2_compare_le(lhs.x1, rhs.x1));
	}

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
		return vec4h(half2_select(cond.x0, a.x0, b.x0), half2_select(cond.x1, a.x1, b.x1));
	}
#else
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec4h operator==(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator!=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator>(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator>=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator<(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator<=(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}

	DEVICE_INLINE vec4h operator+(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator-(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator*(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h operator/(const vec4h &lhs, const vec4h &rhs)
	{
		return vec4h();
	}

	DEVICE_INLINE vec4h abs(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h max(vec4h a, vec4h b)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h min(vec4h a, vec4h b)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h ceil(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h floor(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h sqrt(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h exp(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h log(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h tanh(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h sin(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h cos(vec4h a)
	{
		return vec4h();
	}
	DEVICE_INLINE vec4h erf(const vec4h &a)
	{
		return vec4h();
	}

	DEVICE_INLINE half horizontal_add(vec4h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_max(vec4h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_min(vec4h a)
	{
		return half { };
	}

	DEVICE_INLINE vec4h select(const vec4h &cond, const vec4h &a, const vec4h &b)
	{
		return vec4h();
	}
#endif
} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC4H_CUH_ */
