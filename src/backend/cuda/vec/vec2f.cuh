/*
 * vec2f.cuh
 *
 *  Created on: Jul 29, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC2F_CUH_
#define BACKEND_CUDA_VEC_VEC2F_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors
{
	using vec2f = vec<float, 2>;

	template<>
	class __builtin_align__(8) vec<float, 2>
	{
		public:
			float x0, x1;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit HOST_DEVICE vec(float f0, float f1) :
					x0(f0),
					x1(f1)
			{
			}
			explicit HOST_DEVICE vec(float f) :
					vec(f, f)
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
				*this = reinterpret_cast<const vec2f*>(ptr)[0];
			}
			HOST_DEVICE void store(float *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec2f*>(ptr)[0] = *this;
			}
			HOST_DEVICE void partial_load(const float *__restrict__ ptr, int num)
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 2);
				switch (num)
				{
					case 0:
						break;
					case 1:
						x0 = ptr[0];
						break;
					default:
						x0 = ptr[0];
						x1 = ptr[1];
						break;
				}
			}
			HOST_DEVICE void partial_store(float *__restrict__ ptr, int num) const
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 2);
				switch (num)
				{
					case 0:
						break;
					case 1:
						ptr[0] = x0;
						break;
					default:
						ptr[0] = x0;
						ptr[1] = x1;
						break;
				}
			}
			HOST_DEVICE vec2f operator-() const
			{
				return vec2f(-x0, -x1);
			}
			HOST_DEVICE vec2f operator~() const
			{
				return vec2f(bit_invert(x0), bit_invert(x1));
			}
			HOST_DEVICE vec2f& operator=(float x)
			{
				x0 = x;
				x1 = x;
				return *this;
			}
			HOST_DEVICE int size() const
			{
				return 2;
			}
			HOST_DEVICE float operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				return (idx == 0) ? x0 : x1;
			}
			HOST_DEVICE float& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				return (idx == 0) ? x0 : x1;
			}
	};

	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec2f operator==(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 == rhs.x0), to_mask<float>(lhs.x1 == rhs.x1));
	}
	DEVICE_INLINE vec2f operator!=(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 != rhs.x0), to_mask<float>(lhs.x1 != rhs.x1));
	}
	DEVICE_INLINE vec2f operator>(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 > rhs.x0), to_mask<float>(lhs.x1 > rhs.x1));
	}
	DEVICE_INLINE vec2f operator>=(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 >= rhs.x0), to_mask<float>(lhs.x1 >= rhs.x1));
	}
	DEVICE_INLINE vec2f operator<(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 < rhs.x0), to_mask<float>(lhs.x1 < rhs.x1));
	}
	DEVICE_INLINE vec2f operator<=(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(to_mask<float>(lhs.x0 <= rhs.x0), to_mask<float>(lhs.x1 <= rhs.x1));
	}

	DEVICE_INLINE vec2f operator+(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(lhs.x0 + rhs.x0, lhs.x1 + rhs.x1);
	}
	DEVICE_INLINE vec2f operator-(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(lhs.x0 - rhs.x0, lhs.x1 - rhs.x1);
	}
	DEVICE_INLINE vec2f operator*(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(lhs.x0 * rhs.x0, lhs.x1 * rhs.x1);
	}
	DEVICE_INLINE vec2f operator/(const vec2f &lhs, const vec2f &rhs)
	{
		return vec2f(lhs.x0 / rhs.x0, lhs.x1 / rhs.x1);
	}

	DEVICE_INLINE vec2f abs(const vec2f &a)
	{
		return vec2f(fabsf(a.x0), fabsf(a.x1));
	}
	DEVICE_INLINE vec2f max(const vec2f &a, const vec2f &b)
	{
		return vec2f(fmax(a.x0, b.x0), fmax(a.x1, b.x1));
	}
	DEVICE_INLINE vec2f min(const vec2f &a, const vec2f &b)
	{
		return vec2f(fmin(a.x0, b.x0), fmin(a.x1, b.x1));
	}
	DEVICE_INLINE vec2f ceil(const vec2f &a)
	{
		return vec2f(ceilf(a.x0), ceilf(a.x1));
	}
	DEVICE_INLINE vec2f floor(const vec2f &a)
	{
		return vec2f(floorf(a.x0), floorf(a.x1));
	}
	DEVICE_INLINE vec2f sqrt(const vec2f &a)
	{
		return vec2f(sqrtf(a.x0), sqrtf(a.x1));
	}
	DEVICE_INLINE vec2f pow(const vec2f &a, const vec2f &b)
	{
		return vec2f(powf(a.x0, b.x0), powf(a.x1, b.x1));
	}
	DEVICE_INLINE vec2f mod(const vec2f &a, const vec2f &b)
	{
		return vec2f(fmodf(a.x0, b.x0), fmodf(a.x1, b.x1));
	}
	DEVICE_INLINE vec2f exp(const vec2f &a)
	{
		return vec2f(expf(a.x0), expf(a.x1));
	}
	DEVICE_INLINE vec2f log(const vec2f &a)
	{
		return vec2f(logf(a.x0), logf(a.x1));
	}
	DEVICE_INLINE vec2f tanh(const vec2f &a)
	{
		return vec2f(tanhf(a.x0), tanhf(a.x1));
	}
	DEVICE_INLINE vec2f expm1(const vec2f &a)
	{
		return vec2f(expm1f(a.x0), expm1f(a.x1));
	}
	DEVICE_INLINE vec2f log1p(const vec2f &a)
	{
		return vec2f(log1pf(a.x0), log1pf(a.x1));
	}
	DEVICE_INLINE vec2f sin(const vec2f &a)
	{
		return vec2f(sinf(a.x0), sinf(a.x1));
	}
	DEVICE_INLINE vec2f cos(const vec2f &a)
	{
		return vec2f(cosf(a.x0), cosf(a.x1));
	}
	DEVICE_INLINE vec2f tan(const vec2f &a)
	{
		return vec2f(tanf(a.x0), tanf(a.x1));
	}
	DEVICE_INLINE vec2f erf(const vec2f &a)
	{
		return vec2f(erff(a.x0), erff(a.x1));
	}

	DEVICE_INLINE float horizontal_add(const vec2f &a)
	{
		return a.x0 + a.x1;
	}
	DEVICE_INLINE float horizontal_max(const vec2f &a)
	{
		return fmax(a.x0, a.x1);
	}
	DEVICE_INLINE float horizontal_min(const vec2f &a)
	{
		return fmin(a.x0, a.x1);
	}

	DEVICE_INLINE vec2f select(const vec2f &cond, const vec2f &a, const vec2f &b)
	{
		return vec2f(is_true(cond.x0) ? a.x0 : b.x0, is_true(cond.x1) ? a.x1 : b.x1);
	}

	DEVICE_INLINE void atomic_add(float *address, const vec2f &value)
	{
		atomicAdd(address + 0, value.x0);
		atomicAdd(address + 1, value.x1);
	}

} /* namespace vectors */

#endif /* BACKEND_CUDA_VEC_VEC2F_CUH_ */
