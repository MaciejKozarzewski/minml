/*
 * vec1h.cuh
 *
 *  Created on: Aug 9, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC1H_CUH_
#define BACKEND_CUDA_VEC_VEC1H_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors
{
	using vec1h = vec<half, 1>;

	template<>
	class __builtin_align__(2) vec<half, 1>
	{
		public:
			half x0;

			__device__ vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit __device__ vec(float f) :
					vec1h(static_cast<half>(f))
			{
			}
			explicit __device__ vec(half h) :
					x0(h)
			{
			}
			__device__ vec(const half *__restrict__ ptr)
			{
				load(ptr);
			}
			__device__ void load(const half *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				x0 = ptr[0];
			}
			__device__ void store(half *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				ptr[0] = x0;
			}
			__device__ vec1h operator-() const
			{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
				return vec1h(-x0);
#else
				return vec1h();
#endif
			}
			__device__ vec1h operator~() const
			{
				return vec1h(bit_invert(x0));
			}
			__device__ int size() const
			{
				return 1;
			}
			__device__ half operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				return x0;
			}
			__device__ half& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				return x0;
			}
	};

#if __CUDA_ARCH__ >= FP16_MIN_ARCH
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec1h operator==(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 == rhs.x0));
	}
	DEVICE_INLINE vec1h operator!=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 != rhs.x0));
	}
	DEVICE_INLINE vec1h operator>(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 > rhs.x0));
	}
	DEVICE_INLINE vec1h operator>=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 >= rhs.x0));
	}
	DEVICE_INLINE vec1h operator<(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 < rhs.x0));
	}
	DEVICE_INLINE vec1h operator<=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(to_mask<half>(lhs.x0 < rhs.x0));
	}

	DEVICE_INLINE vec1h operator+(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(lhs.x0 + rhs.x0);
	}
	DEVICE_INLINE vec1h operator-(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(lhs.x0 - rhs.x0);
	}
	DEVICE_INLINE vec1h operator*(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(lhs.x0 * rhs.x0);
	}
	DEVICE_INLINE vec1h operator/(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h(lhs.x0 / rhs.x0);
	}

	DEVICE_INLINE vec1h abs(vec1h a)
	{
		return vec1h(__habs(a.x0));
	}
	DEVICE_INLINE vec1h max(vec1h a, vec1h b)
	{
		return vec1h(__hmax(a.x0, b.x0));
	}
	DEVICE_INLINE vec1h min(vec1h a, vec1h b)
	{
		return vec1h(__hmin(a.x0, b.x0));
	}
	DEVICE_INLINE vec1h ceil(vec1h a)
	{
		return vec1h(hceil(a.x0));
	}
	DEVICE_INLINE vec1h floor(vec1h a)
	{
		return vec1h(hfloor(a.x0));
	}
	DEVICE_INLINE vec1h sqrt(vec1h a)
	{
		return vec1h(hsqrt(a.x0));
	}
	DEVICE_INLINE vec1h exp(vec1h a)
	{
		return vec1h(hexp(a.x0));
	}
	DEVICE_INLINE vec1h log(vec1h a)
	{
		return vec1h(hlog(a.x0));
	}
	DEVICE_INLINE vec1h tanh(vec1h a)
	{
		const vec1h p = exp(a);
		const vec1h m = exp(-a);
		return (p - m) / (p + m);
	}
	DEVICE_INLINE vec1h sin(vec1h a)
	{
		return vec1h(hsin(a.x0));
	}
	DEVICE_INLINE vec1h cos(vec1h a)
	{
		return vec1h(hcos(a.x0));
	}
	DEVICE_INLINE vec1h erf(const vec1h &a)
	{
		return tanh(vec1h(0.797884561f) * a * (vec1h(1.0f) + vec1h(0.044715f) * square(a)));
	}

	DEVICE_INLINE half horizontal_add(vec1h a)
	{
		return a.x0;
	}
	DEVICE_INLINE half horizontal_max(vec1h a)
	{
		return a.x0;
	}
	DEVICE_INLINE half horizontal_min(vec1h a)
	{
		return a.x0;
	}

	DEVICE_INLINE vec1h select(const vec1h &cond, const vec1h &a, const vec1h &b)
	{
		return vec1h(is_true(cond.x0) ? a.x0 : b.x0);
	}
#else
	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec1h operator==(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator!=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator>(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator>=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator<(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator<=(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}

	DEVICE_INLINE vec1h operator+(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator-(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator*(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h operator/(const vec1h &lhs, const vec1h &rhs)
	{
		return vec1h();
	}

	DEVICE_INLINE vec1h abs(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h max(vec1h a, vec1h b)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h min(vec1h a, vec1h b)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h ceil(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h floor(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h sqrt(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h exp(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h log(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h tanh(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h sin(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h cos(vec1h a)
	{
		return vec1h();
	}
	DEVICE_INLINE vec1h erf(const vec1h &a)
	{
		return vec1h();
	}

	DEVICE_INLINE half horizontal_add(vec1h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_max(vec1h a)
	{
		return half { };
	}
	DEVICE_INLINE half horizontal_min(vec1h a)
	{
		return half { };
	}

	DEVICE_INLINE vec1h select(const vec1h &cond, const vec1h &a, const vec1h &b)
	{
		return vec1h();
	}
#endif

} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC1H_CUH_ */
