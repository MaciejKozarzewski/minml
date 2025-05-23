/*
 * vec2h.cuh
 *
 *  Created on: Aug 9, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC2H_CUH_
#define BACKEND_CUDA_VEC_VEC2H_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors
{
	using vec2h = vec<half, 2>;

	template<>
	class __builtin_align__(4) vec<half, 2>
	{
		public:
			half2 x0;

			__device__ vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit __device__ vec(float f) :
					vec2h(static_cast<half>(f))
			{
			}
			explicit __device__ vec(half2 h0) :
					x0(h0)
			{
			}
			explicit __device__ vec(half h) :
					vec2h(h, h)
			{
			}
			explicit __device__ vec(half h0, half h1) :
					x0(h0, h1)
			{
			}
			__device__ vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			__device__ void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				*this = reinterpret_cast<const vec2h*>(ptr)[0];
			}
			__device__ void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec2h*>(ptr)[0] = *this;
			}
			__device__ vec2h operator-() const
			{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
				return vec2h(-x0);
#else
				return vec2h();
#endif
			}
			__device__ vec2h operator~() const
			{
				return vec2h(bit_invert(x0));
			}
			__device__ int size() const
			{
				return 2;
			}
			__device__ half operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				return (idx == 0) ? x0.x : x0.y;
			}
			__device__ half& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				return (idx == 0) ? x0.x : x0.y;
			}
	};

#if __CUDA_ARCH__ >= FP16_MIN_ARCH
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec2h operator==(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_eq(lhs.x0, rhs.x0));
	}
	DEVICE_INLINE vec2h operator!=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_neq(lhs.x0, rhs.x0));
	}
	DEVICE_INLINE vec2h operator>(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_gt(lhs.x0, rhs.x0));
	}
	DEVICE_INLINE vec2h operator>=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_ge(lhs.x0, rhs.x0));
	}
	DEVICE_INLINE vec2h operator<(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_lt(lhs.x0, rhs.x0));
	}
	DEVICE_INLINE vec2h operator<=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(half2_compare_le(lhs.x0, rhs.x0));
	}

	DEVICE_INLINE vec2h operator+(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(lhs.x0 + rhs.x0);
	}
	DEVICE_INLINE vec2h operator-(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(lhs.x0 - rhs.x0);
	}
	DEVICE_INLINE vec2h operator*(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(lhs.x0 * rhs.x0);
	}
	DEVICE_INLINE vec2h operator/(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h(lhs.x0 / rhs.x0);
	}

	DEVICE_INLINE vec2h abs(vec2h a)
	{
		return vec2h(__habs2(a.x0));
	}
	DEVICE_INLINE vec2h max(vec2h a, vec2h b)
	{
		return vec2h(__hmax2(a.x0, b.x0));
	}
	DEVICE_INLINE vec2h min(vec2h a, vec2h b)
	{
		return vec2h(__hmin2(a.x0, b.x0));
	}
	DEVICE_INLINE vec2h ceil(vec2h a)
	{
		return vec2h(h2ceil(a.x0));
	}
	DEVICE_INLINE vec2h floor(vec2h a)
	{
		return vec2h(h2floor(a.x0));
	}
	DEVICE_INLINE vec2h sqrt(vec2h a)
	{
		return vec2h(h2sqrt(a.x0));
	}
	DEVICE_INLINE vec2h exp(vec2h a)
	{
		return vec2h(h2exp(a.x0));
	}
	DEVICE_INLINE vec2h log(vec2h a)
	{
		return vec2h(h2log(a.x0));
	}
	DEVICE_INLINE vec2h tanh(vec2h a)
	{
		const vec2h p = exp(a);
		const vec2h m = exp(-a);
		return (p - m) / (p + m);
	}
	DEVICE_INLINE vec2h sin(vec2h a)
	{
		return vec2h(h2sin(a.x0));
	}
	DEVICE_INLINE vec2h cos(vec2h a)
	{
		return vec2h(h2cos(a.x0));
	}
	DEVICE_INLINE vec2h erf(const vec2h &a)
	{
		return tanh(vec2h(0.797884561f) * a * (vec2h(1.0f) + vec2h(0.044715f) * square(a)));
	}

	DEVICE_INLINE half horizontal_add(vec2h a)
	{
		return a.x0.x + a.x0.y;
	}
	DEVICE_INLINE half horizontal_max(vec2h a)
	{
		return __hmax(a.x0.x, a.x0.y);
	}
	DEVICE_INLINE half horizontal_min(vec2h a)
	{
		return __hmin(a.x0.x, a.x0.y);
	}

	DEVICE_INLINE vec2h select(const vec2h &cond, const vec2h &a, const vec2h &b)
	{
		return vec2h(half2_select(cond.x0, a.x0, b.x0));
	}
#else
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec2h operator==(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator!=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator>(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator>=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator<(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator<=(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}

	DEVICE_INLINE vec2h operator+(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator-(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator*(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h operator/(const vec2h &lhs, const vec2h &rhs)
	{
		return vec2h();
	}

	DEVICE_INLINE vec2h abs(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h max(vec2h a, vec2h b)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h min(vec2h a, vec2h b)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h ceil(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h floor(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h sqrt(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h exp(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h log(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h tanh(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h sin(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h cos(vec2h a)
	{
		return vec2h();
	}
	DEVICE_INLINE vec2h erf(const vec2h &a)
	{
		return vec2h();
	}

	DEVICE_INLINE half horizontal_add(vec2h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_max(vec2h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_min(vec2h a)
	{
		return half { };
	}

	DEVICE_INLINE vec2h select(const vec2h &cond, const vec2h &a, const vec2h &b)
	{
		return vec2h();
	}
#endif
} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC2H_CUH_ */
