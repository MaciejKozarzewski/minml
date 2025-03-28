/*
 * vec4f.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_VEC4F_CUH_
#define BACKEND_CUDA_VEC_VEC4F_CUH_

#include "generic_vec.cuh"
#include "utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors
{
	using vec4f = vec<float, 4>;

	template<>
	class __builtin_align__(16) vec<float, 4>
	{
		public:
			float x0, x1, x2, x3;

			HOST_DEVICE vec() // @suppress("Class members should be properly initialized")
			{
			}
			explicit HOST_DEVICE vec(float f0, float f1, float f2, float f3) :
					x0(f0),
					x1(f1),
					x2(f2),
					x3(f3)
			{
			}
			explicit HOST_DEVICE vec(float f) :
					vec(f, f, f, f)
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
				*this = reinterpret_cast<const vec4f*>(ptr)[0];
			}
			HOST_DEVICE void store(float *__restrict__ ptr) const
			{
				assert(ptr != nullptr);
				reinterpret_cast<vec4f*>(ptr)[0] = *this;
			}
			HOST_DEVICE void partial_load(const float *__restrict__ ptr, int num)
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 4);
				switch (num)
				{
					case 0:
						break;
					case 1:
						x0 = ptr[0];
						break;
					case 2:
						x0 = ptr[0];
						x1 = ptr[1];
						break;
					case 3:
						x0 = ptr[0];
						x1 = ptr[1];
						x2 = ptr[2];
						break;
					default:
						x0 = ptr[0];
						x1 = ptr[1];
						x2 = ptr[2];
						x3 = ptr[3];
						break;
				}
			}
			HOST_DEVICE void partial_store(float *__restrict__ ptr, int num) const
			{
				assert(ptr != nullptr);
				assert(0 <= num && num <= 4);
				switch (num)
				{
					case 0:
						break;
					case 1:
						ptr[0] = x0;
						break;
					case 2:
						ptr[0] = x0;
						ptr[1] = x1;
						break;
					case 3:
						ptr[0] = x0;
						ptr[1] = x1;
						ptr[2] = x2;
						break;
					default:
						ptr[0] = x0;
						ptr[1] = x1;
						ptr[2] = x2;
						ptr[3] = x3;
						break;
				}
			}
			HOST_DEVICE vec4f operator-() const
			{
				return vec4f(-x0, -x1, -x2, -x3);
			}
			HOST_DEVICE vec4f operator~() const
			{
				return vec4f(bit_invert(x0), bit_invert(x1), bit_invert(x2), bit_invert(x3));
			}
			HOST_DEVICE vec4f& operator=(float x)
			{
				x0 = x;
				x1 = x;
				x2 = x;
				x3 = x;
				return *this;
			}
			HOST_DEVICE int size() const
			{
				return 4;
			}
			HOST_DEVICE float operator[](int idx) const
			{
				assert(0 <= idx && idx < size());
				switch (idx)
				{
					default:
					case 0:
						return x0;
					case 1:
						return x1;
					case 2:
						return x2;
					case 3:
						return x3;
				}
			}
			HOST_DEVICE float& operator[](int idx)
			{
				assert(0 <= idx && idx < size());
				switch (idx)
				{
					default:
					case 0:
						return x0;
					case 1:
						return x1;
					case 2:
						return x2;
					case 3:
						return x3;
				}
			}
	};

	/*
	 * comparison operators
	 */
	DEVICE_INLINE vec4f operator==(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 == rhs.x0), to_mask<float>(lhs.x1 == rhs.x1), to_mask<float>(lhs.x2 == rhs.x2),
				to_mask<float>(lhs.x3 == rhs.x3));
	}
	DEVICE_INLINE vec4f operator!=(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 != rhs.x0), to_mask<float>(lhs.x1 != rhs.x1), to_mask<float>(lhs.x2 != rhs.x2),
				to_mask<float>(lhs.x3 != rhs.x3));
	}
	DEVICE_INLINE vec4f operator>(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 > rhs.x0), to_mask<float>(lhs.x1 > rhs.x1), to_mask<float>(lhs.x2 > rhs.x2),
				to_mask<float>(lhs.x3 > rhs.x3));
	}
	DEVICE_INLINE vec4f operator>=(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 >= rhs.x0), to_mask<float>(lhs.x1 >= rhs.x1), to_mask<float>(lhs.x2 >= rhs.x2),
				to_mask<float>(lhs.x3 >= rhs.x3));
	}
	DEVICE_INLINE vec4f operator<(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 < rhs.x0), to_mask<float>(lhs.x1 < rhs.x1), to_mask<float>(lhs.x2 < rhs.x2),
				to_mask<float>(lhs.x3 < rhs.x3));
	}
	DEVICE_INLINE vec4f operator<=(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(to_mask<float>(lhs.x0 < rhs.x0), to_mask<float>(lhs.x1 < rhs.x1), to_mask<float>(lhs.x2 < rhs.x2),
				to_mask<float>(lhs.x3 < rhs.x3));
	}

	DEVICE_INLINE vec4f operator+(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(lhs.x0 + rhs.x0, lhs.x1 + rhs.x1, lhs.x2 + rhs.x2, lhs.x3 + rhs.x3);
	}
	DEVICE_INLINE vec4f operator-(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(lhs.x0 - rhs.x0, lhs.x1 - rhs.x1, lhs.x2 - rhs.x2, lhs.x3 - rhs.x3);
	}
	DEVICE_INLINE vec4f operator*(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(lhs.x0 * rhs.x0, lhs.x1 * rhs.x1, lhs.x2 * rhs.x2, lhs.x3 * rhs.x3);
	}
	DEVICE_INLINE vec4f operator/(const vec4f &lhs, const vec4f &rhs)
	{
		return vec4f(lhs.x0 / rhs.x0, lhs.x1 / rhs.x1, lhs.x2 / rhs.x2, lhs.x3 / rhs.x3);
	}

	DEVICE_INLINE vec4f abs(const vec4f &a)
	{
		return vec4f(fabsf(a.x0), fabsf(a.x1), fabsf(a.x2), fabsf(a.x3));
	}
	DEVICE_INLINE vec4f max(const vec4f &a, const vec4f &b)
	{
		return vec4f(fmax(a.x0, b.x0), fmax(a.x1, b.x1), fmax(a.x2, b.x2), fmax(a.x3, b.x3));
	}
	DEVICE_INLINE vec4f min(const vec4f &a, const vec4f &b)
	{
		return vec4f(fmin(a.x0, b.x0), fmin(a.x1, b.x1), fmin(a.x2, b.x2), fmin(a.x3, b.x3));
	}
	DEVICE_INLINE vec4f ceil(const vec4f &a)
	{
		return vec4f(ceilf(a.x0), ceilf(a.x1), ceilf(a.x2), ceilf(a.x3));
	}
	DEVICE_INLINE vec4f floor(const vec4f &a)
	{
		return vec4f(floorf(a.x0), floorf(a.x1), floorf(a.x2), floorf(a.x3));
	}
	DEVICE_INLINE vec4f sqrt(const vec4f &a)
	{
		return vec4f(sqrtf(a.x0), sqrtf(a.x1), sqrtf(a.x2), sqrtf(a.x3));
	}
	DEVICE_INLINE vec4f pow(const vec4f &a, const vec4f &b)
	{
		return vec4f(powf(a.x0, b.x0), powf(a.x1, b.x1), powf(a.x2, b.x2), powf(a.x3, b.x3));
	}
	DEVICE_INLINE vec4f mod(const vec4f &a, const vec4f &b)
	{
		return vec4f(fmodf(a.x0, b.x0), fmodf(a.x1, b.x1), fmodf(a.x2, b.x2), fmodf(a.x3, b.x3));
	}
	DEVICE_INLINE vec4f exp(const vec4f &a)
	{
		return vec4f(expf(a.x0), expf(a.x1), expf(a.x2), expf(a.x3));
	}
	DEVICE_INLINE vec4f log(const vec4f &a)
	{
		return vec4f(logf(a.x0), logf(a.x1), logf(a.x2), logf(a.x3));
	}
	DEVICE_INLINE vec4f tanh(const vec4f &a)
	{
		return vec4f(tanhf(a.x0), tanhf(a.x1), tanhf(a.x2), tanhf(a.x3));
	}
	DEVICE_INLINE vec4f expm1(const vec4f &a)
	{
		return vec4f(expm1f(a.x0), expm1f(a.x1), expm1f(a.x2), expm1f(a.x3));
	}
	DEVICE_INLINE vec4f log1p(const vec4f &a)
	{
		return vec4f(log1pf(a.x0), log1pf(a.x1), log1pf(a.x2), log1pf(a.x3));
	}
	DEVICE_INLINE vec4f sin(const vec4f &a)
	{
		return vec4f(sinf(a.x0), sinf(a.x1), sinf(a.x2), sinf(a.x3));
	}
	DEVICE_INLINE vec4f cos(const vec4f &a)
	{
		return vec4f(cosf(a.x0), cosf(a.x1), cosf(a.x2), cosf(a.x3));
	}
	DEVICE_INLINE vec4f tan(const vec4f &a)
	{
		return vec4f(tanf(a.x0), tanf(a.x1), tanf(a.x2), tanf(a.x3));
	}
	DEVICE_INLINE vec4f erf(const vec4f &a)
	{
		return vec4f(erff(a.x0), erff(a.x1), erff(a.x2), erff(a.x3));
	}

	DEVICE_INLINE float horizontal_add(const vec4f &a)
	{
		return a.x0 + a.x1 + a.x2 + a.x3;
	}
	DEVICE_INLINE float horizontal_max(const vec4f &a)
	{
		return fmax(fmax(a.x0, a.x1), fmax(a.x2, a.x3));
	}
	DEVICE_INLINE float horizontal_min(const vec4f &a)
	{
		return fmin(fmin(a.x0, a.x1), fmin(a.x2, a.x3));
	}

	DEVICE_INLINE vec4f select(const vec4f &cond, const vec4f &a, const vec4f &b)
	{
		return vec4f(is_true(cond.x0) ? a.x0 : b.x0, is_true(cond.x1) ? a.x1 : b.x1, is_true(cond.x2) ? a.x2 : b.x2, is_true(cond.x3) ? a.x3 : b.x3);
	}

	DEVICE_INLINE void atomic_add(float *address, const vec4f &value)
	{
		atomicAdd(address + 0, value.x0);
		atomicAdd(address + 1, value.x1);
		atomicAdd(address + 2, value.x2);
		atomicAdd(address + 3, value.x3);
	}

} /* namespace vectors */

#endif /* BACKEND_CUDA_VECTORS_VEC4F_CUH_ */
