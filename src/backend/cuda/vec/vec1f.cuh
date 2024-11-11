/*
 * vec1f.cuh
 *
 *  Created on: Jul 29, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC1F_CUH_
#define BACKEND_CUDA_VEC_VEC1F_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors2
{
	using vec1f = vec<float, 1>;

	template<>
	class __builtin_align__(4) vec<float, 1>
	{
		public:
			float x0;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			HOST_DEVICE vec(float f) :
					x0(f)
			{
			}
			HOST_DEVICE vec(const float *__restrict__ ptr)
			{
				load(ptr);
			}
			HOST_DEVICE vec(const float *__restrict__ ptr, int num)
			{
				partial_load(ptr, num);
			}
			HOST_DEVICE void load(const float *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				x0 = ptr[0];
			}
			HOST_DEVICE void store(float *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				ptr[0] = x0;
			}
			HOST_DEVICE void partial_load(const float *__restrict__ ptr, int num)
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 1);
				if (num == 1)
					x0 = ptr[0];
			}
			HOST_DEVICE void partial_store(float *__restrict__ ptr, int num) const
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 1);
				if (num == 1)
					ptr[0] = x0;
			}
			HOST_DEVICE vec1f operator-() const
			{
				return vec1f(-x0);
			}
			HOST_DEVICE vec1f operator~() const
			{
				return vec1f(bit_invert(x0));
			}
			HOST_DEVICE vec1f& operator=(float x)
			{
				x0 = x;
				return *this;
			}
			HOST_DEVICE int size() const
			{
				return 1;
			}
	};

	DEVICE_INLINE vec1f operator+(const vec1f &lhs, const vec1f &rhs)
	{
		return vec1f(lhs.x0 + rhs.x0);
	}
	DEVICE_INLINE vec1f operator-(const vec1f &lhs, const vec1f &rhs)
	{
		return vec1f(lhs.x0 - rhs.x0);
	}
	DEVICE_INLINE vec1f operator*(const vec1f &lhs, const vec1f &rhs)
	{
		return vec1f(lhs.x0 * rhs.x0);
	}
	DEVICE_INLINE vec1f operator/(const vec1f &lhs, const vec1f &rhs)
	{
		return vec1f(lhs.x0 / rhs.x0);
	}

	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec1f operator==(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 == rhs.x0);
	}
	DEVICE_INLINE vec1f operator!=(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 != rhs.x0);
	}
	DEVICE_INLINE vec1f operator>(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 > rhs.x0);
	}
	DEVICE_INLINE vec1f operator>=(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 >= rhs.x0);
	}
	DEVICE_INLINE vec1f operator<(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 < rhs.x0);
	}
	DEVICE_INLINE vec1f operator<=(const vec1f &lhs, const vec1f &rhs)
	{
		return to_mask<float>(lhs.x0 < rhs.x0);
	}

	DEVICE_INLINE vec1f abs(const vec1f &a)
	{
		return vec1f(fabsf(a.x0));
	}
	DEVICE_INLINE vec1f max(const vec1f &a, const vec1f &b)
	{
		return vec1f(fmax(a.x0, b.x0));
	}
	DEVICE_INLINE vec1f min(const vec1f &a, const vec1f &b)
	{
		return vec1f(fmin(a.x0, b.x0));
	}
	DEVICE_INLINE vec1f ceil(const vec1f &a)
	{
		return vec1f(ceilf(a.x0));
	}
	DEVICE_INLINE vec1f floor(const vec1f &a)
	{
		return vec1f(floorf(a.x0));
	}
	DEVICE_INLINE vec1f sqrt(const vec1f &a)
	{
		return vec1f(sqrtf(a.x0));
	}
	DEVICE_INLINE vec1f pow(const vec1f &a, const vec1f &b)
	{
		return vec1f(powf(a.x0, b.x0));
	}
	DEVICE_INLINE vec1f mod(const vec1f &a, const vec1f &b)
	{
		return vec1f(fmodf(a.x0, b.x0));
	}
	DEVICE_INLINE vec1f exp(const vec1f &a)
	{
		return vec1f(expf(a.x0));
	}
	DEVICE_INLINE vec1f log(const vec1f &a)
	{
		return vec1f(logf(a.x0));
	}
	DEVICE_INLINE vec1f tanh(const vec1f &a)
	{
		return vec1f(tanhf(a.x0));
	}
	DEVICE_INLINE vec1f expm1(const vec1f &a)
	{
		return vec1f(expm1f(a.x0));
	}
	DEVICE_INLINE vec1f log1p(const vec1f &a)
	{
		return vec1f(log1pf(a.x0));
	}
	DEVICE_INLINE vec1f sin(const vec1f &a)
	{
		return vec1f(sinf(a.x0));
	}
	DEVICE_INLINE vec1f cos(const vec1f &a)
	{
		return vec1f(cosf(a.x0));
	}
	DEVICE_INLINE vec1f tan(const vec1f &a)
	{
		return vec1f(tanf(a.x0));
	}
	DEVICE_INLINE vec1f erf(const vec1f &a)
	{
		return vec1f(erff(a.x0));
	}

	DEVICE_INLINE float horizontal_add(const vec1f &a)
	{
		return a.x0;
	}
	DEVICE_INLINE float horizontal_max(const vec1f &a)
	{
		return a.x0;
	}
	DEVICE_INLINE float horizontal_min(const vec1f &a)
	{
		return a.x0;
	}

	DEVICE_INLINE vec1f select(const vec1f &cond, const vec1f &a, const vec1f &b)
	{
		return vec1f(is_true(cond.x0) ? a.x0 : b.x0);
	}

	DEVICE_INLINE void atomic_add(float *address, const vec1f &value)
	{
		atomicAdd(address + 0, value.x0);
	}

} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC1F_CUH_ */
