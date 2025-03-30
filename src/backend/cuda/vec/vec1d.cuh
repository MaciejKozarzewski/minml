/*
 * vec1d.cuh
 *
 *  Created on: Nov 3, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC1D_CUH_
#define BACKEND_CUDA_VEC_VEC1D_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>

namespace vectors
{
	using vec1d = vec<double, 1>;

	template<>
	class __builtin_align__(8) vec<double, 1>
	{
		public:
			double x0;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit HOST_DEVICE vec(double d) :
					x0(d)
			{
			}
			HOST_DEVICE vec(const double *__restrict__ ptr)
			{
				load(ptr);
			}
			HOST_DEVICE vec(const double *__restrict__ ptr, int num)
			{
				partial_load(ptr, num);
			}
			HOST_DEVICE void load(const double *__restrict__ ptr)
			{
				assert(ptr != nullptr);
				x0 = ptr[0];
			}
			HOST_DEVICE void store(double *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				ptr[0] = x0;
			}
			HOST_DEVICE void partial_load(const double *__restrict__ ptr, int num)
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 1);
				if (num == 1)
					x0 = ptr[0];
			}
			HOST_DEVICE void partial_store(double *__restrict__ ptr, int num) const
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 1);
				if (num == 1)
					ptr[0] = x0;
			}
			HOST_DEVICE vec1d operator-() const
			{
				return vec1d(-x0);
			}
			HOST_DEVICE vec1d operator~() const
			{
				return vec1d(bit_invert(x0));
			}
			HOST_DEVICE vec1d& operator=(float x)
			{
				x0 = x;
				return *this;
			}
			HOST_DEVICE int size() const
			{
				return 1;
			}
			HOST_DEVICE double operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				return x0;
			}
			HOST_DEVICE double& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				return x0;
			}
	};

	DEVICE_INLINE vec1d operator+(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(lhs.x0 + rhs.x0);
	}
	DEVICE_INLINE vec1d operator-(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(lhs.x0 - rhs.x0);
	}
	DEVICE_INLINE vec1d operator*(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(lhs.x0 * rhs.x0);
	}
	DEVICE_INLINE vec1d operator/(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(lhs.x0 / rhs.x0);
	}

	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec1d operator==(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 == rhs.x0));
	}
	DEVICE_INLINE vec1d operator!=(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 != rhs.x0));
	}
	DEVICE_INLINE vec1d operator>(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 > rhs.x0));
	}
	DEVICE_INLINE vec1d operator>=(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 >= rhs.x0));
	}
	DEVICE_INLINE vec1d operator<(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 < rhs.x0));
	}
	DEVICE_INLINE vec1d operator<=(const vec1d &lhs, const vec1d &rhs)
	{
		return vec1d(to_mask<double>(lhs.x0 <= rhs.x0));
	}

	DEVICE_INLINE vec1d abs(const vec1d &a)
	{
		return vec1d(fabs(a.x0));
	}
	DEVICE_INLINE vec1d max(const vec1d &a, const vec1d &b)
	{
		return vec1d(fmax(a.x0, b.x0));
	}
	DEVICE_INLINE vec1d min(const vec1d &a, const vec1d &b)
	{
		return vec1d(fmin(a.x0, b.x0));
	}
	DEVICE_INLINE vec1d ceil(const vec1d &a)
	{
		return vec1d(ceilf(a.x0));
	}
	DEVICE_INLINE vec1d floor(const vec1d &a)
	{
		return vec1d(floorf(a.x0));
	}
	DEVICE_INLINE vec1d sqrt(const vec1d &a)
	{
		return vec1d(sqrtf(a.x0));
	}
	DEVICE_INLINE vec1d pow(const vec1d &a, const vec1d &b)
	{
		return vec1d(powf(a.x0, b.x0));
	}
	DEVICE_INLINE vec1d mod(const vec1d &a, const vec1d &b)
	{
		return vec1d(fmodf(a.x0, b.x0));
	}
	DEVICE_INLINE vec1d exp(const vec1d &a)
	{
		return vec1d(expf(a.x0));
	}
	DEVICE_INLINE vec1d log(const vec1d &a)
	{
		return vec1d(logf(a.x0));
	}
	DEVICE_INLINE vec1d tanh(const vec1d &a)
	{
		return vec1d(tanhf(a.x0));
	}
	DEVICE_INLINE vec1d expm1(const vec1d &a)
	{
		return vec1d(expm1f(a.x0));
	}
	DEVICE_INLINE vec1d log1p(const vec1d &a)
	{
		return vec1d(log1pf(a.x0));
	}
	DEVICE_INLINE vec1d sin(const vec1d &a)
	{
		return vec1d(sinf(a.x0));
	}
	DEVICE_INLINE vec1d cos(const vec1d &a)
	{
		return vec1d(cosf(a.x0));
	}
	DEVICE_INLINE vec1d tan(const vec1d &a)
	{
		return vec1d(tanf(a.x0));
	}
	DEVICE_INLINE vec1d erf(const vec1d &a)
	{
		return vec1d(erff(a.x0));
	}

	DEVICE_INLINE double horizontal_add(const vec1d &a)
	{
		return a.x0;
	}
	DEVICE_INLINE double horizontal_max(const vec1d &a)
	{
		return a.x0;
	}
	DEVICE_INLINE double horizontal_min(const vec1d &a)
	{
		return a.x0;
	}

	DEVICE_INLINE vec1d select(const vec1d &cond, const vec1d &a, const vec1d &b)
	{
		return vec1d(is_true(cond.x0) ? a.x0 : b.x0);
	}

	DEVICE_INLINE void atomic_add(double *address, const vec1d &value)
	{
#if __CUDA_ARCH__ >= 600
		atomicAdd(address + 0, value.x0);
#endif
	}

} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC1D_CUH_ */
