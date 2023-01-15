/*
 * fp32_vector.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef CUDA_FP32_VECTOR_CUH_
#define CUDA_FP32_VECTOR_CUH_

#include "generic_vector.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cassert>

namespace vectors
{
	template<>
	class Vector<float>
	{
		private:
			float m_data;
		public:
			__device__ Vector() // @suppress("Class members should be properly initialized")
			{
			}
			__device__ Vector(half x) :
					m_data(static_cast<float>(x))
			{
			}
			__device__ Vector(float x) :
					m_data(x)
			{
			}
			__device__ Vector(double x) :
					m_data(static_cast<float>(x))
			{
			}
			__device__ Vector(const half *__restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ Vector(const float *__restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ Vector(const double *__restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ void load(const half *__restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = static_cast<float>(ptr[0]);
				else
					m_data = 0.0f;
			}
			__device__ void load(const float *__restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = ptr[0];
				else
					m_data = 0.0f;
			}
			__device__ void load(const double *__restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = static_cast<float>(ptr[0]);
				else
					m_data = 0.0f;
			}
			__device__ void store(half *__restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			__device__ void store(float *__restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			__device__ void store(double *__restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = static_cast<double>(m_data);
			}
			__device__ operator float() const
			{
				return m_data;
			}
			__device__ Vector<float> operator-() const
			{
				return Vector<float>(-m_data);
			}
			__device__ Vector<float> operator~() const
			{
				const uint32_t tmp = ~reinterpret_cast<const uint32_t*>(&m_data)[0];
				return Vector<float>(reinterpret_cast<const float*>(&tmp)[0]);
			}
	};

	template<>
	DEVICE_INLINE Vector<float> vector_zero()
	{
		return Vector<float>(0.0f);
	}
	template<>
	DEVICE_INLINE Vector<float> vector_one()
	{
		return Vector<float>(1.0f);
	}
	template<>
	DEVICE_INLINE Vector<float> vector_epsilon()
	{
		return Vector<float>(1.1920928955078125e-7f);
	}

	DEVICE_INLINE Vector<float> operator+(const Vector<float> &lhs, const Vector<float> &rhs)
	{
		return Vector<float>(static_cast<float>(lhs) + static_cast<float>(rhs));
	}
	DEVICE_INLINE Vector<float> operator-(const Vector<float> &lhs, const Vector<float> &rhs)
	{
		return Vector<float>(static_cast<float>(lhs) - static_cast<float>(rhs));
	}
	DEVICE_INLINE Vector<float> operator*(const Vector<float> &lhs, const Vector<float> &rhs)
	{
		return Vector<float>(static_cast<float>(lhs) * static_cast<float>(rhs));
	}
	DEVICE_INLINE Vector<float> operator/(const Vector<float> &lhs, const Vector<float> &rhs)
	{
		return Vector<float>(static_cast<float>(lhs) / static_cast<float>(rhs));
	}

	DEVICE_INLINE Vector<float> sgn(Vector<float> x) noexcept
	{
		return internal::sgn(static_cast<float>(x));
	}
	DEVICE_INLINE Vector<float> abs(Vector<float> x) noexcept
	{
		return fabsf(x);
	}
	DEVICE_INLINE Vector<float> max(Vector<float> x, Vector<float> y)
	{
		return fmax(x, y);
	}
	DEVICE_INLINE Vector<float> min(Vector<float> x, Vector<float> y)
	{
		return fmin(x, y);
	}
	DEVICE_INLINE Vector<float> ceil(Vector<float> x)
	{
		return ceilf(x);
	}
	DEVICE_INLINE Vector<float> floor(Vector<float> x)
	{
		return floorf(x);
	}
	DEVICE_INLINE Vector<float> sqrt(Vector<float> x)
	{
		return sqrtf(x);
	}
	DEVICE_INLINE Vector<float> pow(Vector<float> x, Vector<float> y)
	{
		return powf(x, y);
	}
	DEVICE_INLINE Vector<float> mod(Vector<float> x, Vector<float> y)
	{
		return fmodf(x, y);
	}
	DEVICE_INLINE Vector<float> exp(Vector<float> x)
	{
		return expf(x);
	}
	DEVICE_INLINE Vector<float> log(Vector<float> x)
	{
		return logf(x);
	}
	DEVICE_INLINE Vector<float> tanh(Vector<float> x)
	{
		return tanhf(x);
	}
	DEVICE_INLINE Vector<float> expm1(Vector<float> x)
	{
		return expm1f(x);
	}
	DEVICE_INLINE Vector<float> log1p(Vector<float> x)
	{
		return log1pf(x);
	}
	DEVICE_INLINE Vector<float> sin(Vector<float> x)
	{
		return sinf(x);
	}
	DEVICE_INLINE Vector<float> cos(Vector<float> x)
	{
		return cosf(x);
	}
	DEVICE_INLINE Vector<float> tan(Vector<float> x)
	{
		return tanf(x);
	}

	DEVICE_INLINE float horizontal_add(Vector<float> x)
	{
		return static_cast<float>(x);
	}
	DEVICE_INLINE float horizontal_mul(Vector<float> x)
	{
		return static_cast<float>(x);
	}
	DEVICE_INLINE float horizontal_max(Vector<float> x)
	{
		return static_cast<float>(x);
	}
	DEVICE_INLINE float horizontal_min(Vector<float> x)
	{
		return static_cast<float>(x);
	}
	DEVICE_INLINE float horizontal_or(Vector<float> x)
	{
		return static_cast<float>(x);
	}
	DEVICE_INLINE float horizontal_and(Vector<float> x)
	{
		return static_cast<float>(x);
	}

} /* namespace vectors */

#endif /* CUDA_FP32_VECTOR_CUH_ */
