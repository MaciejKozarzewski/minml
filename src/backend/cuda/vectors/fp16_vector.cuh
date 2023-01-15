/*
 * fp16_vector.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef CUDA_FP16_VECTOR_CUH_
#define CUDA_FP16_VECTOR_CUH_

#include "generic_vector.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cassert>

namespace vectors
{
	template<>
	class Vector<half>
	{
		private:
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
			half2 m_data;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
			float m_data;
#else
#endif
		public:
			__device__ Vector() // @suppress("Class members should be properly initialized")
			{
			}
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
			__device__ Vector(half2 x) :
					m_data(x)
			{
			}
			__device__ Vector(half x) :
					m_data(x, x)
			{
			}
			__device__ Vector(half x, half y) :
					m_data(x, y)
			{
			}
			__device__ Vector(float x) :
					m_data(x, x)
			{
			}
			__device__ Vector(float x, float y) :
					m_data(x, y)
			{
			}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
			__device__ Vector(half x) :
					m_data(x)
			{
			}
			__device__ Vector(float x) :
					m_data(x)
			{
			}
#else
			__device__ Vector(half x)
			{
			}
			__device__ Vector(float x)
			{
			}
			__device__ Vector(double x)
			{
			}
#endif
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
			__device__ Vector(const half * __restrict__ ptr, int num = 2)
			{
				load(ptr, num);
			}
			__device__ Vector(const float * __restrict__ ptr, int num = 2)
			{
				load(ptr, num);
			}
			__device__ void load(const half * __restrict__ ptr, int num = 2)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 2)
				{
					if (is_aligned<half2>(ptr))
						m_data = reinterpret_cast<const half2*>(ptr)[0];
					else
						m_data = half2(ptr[0], ptr[1]);
				}
				else
				{
					if (num == 1)
						m_data = half2(ptr[0], 0.0f);
					else
						m_data = half2(0.0f, 0.0f);
				}
			}
			__device__ void load(const float * __restrict__ ptr, int num = 2)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 2)
					m_data = half2(ptr[0], ptr[1]);
				else
				{
					if (num == 1)
						m_data = half2(ptr[0], 0.0f);
					else
						m_data = half2(0.0f, 0.0f);
				}
			}
			__device__ void store(half * __restrict__ ptr, int num = 2) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 2)
				{
					if (is_aligned<half2>(ptr))
						reinterpret_cast<half2*>(ptr)[0] = m_data;
					else
					{
						ptr[0] = m_data.x;
						ptr[1] = m_data.y;
					}
				}
				else
				{
					if (num == 1)
						ptr[0] = m_data.x;
				}
			}
			__device__ void store(float * __restrict__ ptr, int num = 2) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 2)
				{
					ptr[0] = static_cast<float>(m_data.x);
					ptr[1] = static_cast<float>(m_data.y);
				}
				else
				{
					if (num == 1)
						ptr[0] = static_cast<float>(m_data.x);
				}
			}
			__device__ operator half2() const
			{
				return m_data;
			}
			__device__ half low() const
			{
				return m_data.x;
			}
			__device__ half high() const
			{
				return m_data.y;
			}
			__device__ Vector<half> operator-() const
			{
				return Vector<half>(-m_data);
			}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
			__device__ Vector(const half * __restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ Vector(const float * __restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ void load(const half * __restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = static_cast<float>(ptr[0]);
			}
			__device__ void load(const float * __restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = ptr[0];
			}
			__device__ void store(half * __restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			__device__ void store(float * __restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			__device__ operator float() const
			{
				return m_data;
			}
			__device__ Vector<half> operator-() const
			{
				return Vector<half>(-m_data);
			}
			__device__ half get() const
			{
				return m_data;
			}
#else
			__device__ Vector(const half * __restrict__ ptr, int num = 0)
			{
			}
			__device__ Vector(const float * __restrict__ ptr, int num = 0)
			{
			}
			__device__ void load(const half * __restrict__ ptr, int num = 0)
			{
			}
			__device__ void load(const float * __restrict__ ptr, int num = 0)
			{
			}
			__device__ void store(half * __restrict__ ptr, int num = 0) const
			{
			}
			__device__ void store(float * __restrict__ ptr, int num = 0) const
			{
			}
			__device__ operator float() const
			{
				return 0.0f;
			}
			__device__ Vector<half> operator-() const
			{
				return Vector<half>(0.0f);
			}
#endif
			__device__ Vector<half> operator~() const
			{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
				const uint32_t tmp = ~reinterpret_cast<const uint32_t*>(&m_data)[0];
				return Vector<half>(reinterpret_cast<const half*>(&tmp)[0]);
#else
				return Vector<half>();
#endif
			}
	};

	template<>
	DEVICE_INLINE constexpr int vector_length<half>()
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return 2;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return 1;
#else
		return 0;
#endif
	}

	template<>
	DEVICE_INLINE Vector<half> vector_zero()
	{
		return Vector<half>(0.0f);
	}
	template<>
	DEVICE_INLINE Vector<half> vector_one()
	{
		return Vector<half>(1.0f);
	}
	template<>
	DEVICE_INLINE Vector<half> vector_epsilon()
	{
		return Vector<half>(0.00006103515625f);
	}

	DEVICE_INLINE Vector<half> operator+(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(static_cast<half2>(lhs) + static_cast<half2>(rhs));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) + static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> operator-(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(static_cast<half2>(lhs) - static_cast<half2>(rhs));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) - static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> operator*(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(static_cast<half2>(lhs) * static_cast<half2>(rhs));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) * static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> operator/(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(static_cast<half2>(lhs) / static_cast<half2>(rhs));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) / static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}

	DEVICE_INLINE Vector<half> sgn(Vector<half> x) noexcept
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		half2 tmp = x;
		half2 result;
		result.x = static_cast<half>(internal::sgn(static_cast<float>(tmp.x)));
		result.y = static_cast<half>(internal::sgn(static_cast<float>(tmp.y)));
		return result;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(internal::sgn(static_cast<float>(x)));
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> abs(Vector<half> x) noexcept
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		half2 tmp = x;
		half2 result;
		result.x = static_cast<half>(fabsf(static_cast<float>(tmp.x)));
		result.y = static_cast<half>(fabsf(static_cast<float>(tmp.y)));
		return result;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(fabsf(static_cast<float>(x)));
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> max(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(fmax(static_cast<float>(x.low()), static_cast<float>(y.low())), fmax(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmax(x, y);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> min(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(fmin(static_cast<float>(x.low()), static_cast<float>(y.low())), fmin(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmin(x, y);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> ceil(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(ceilf(static_cast<float>(x.low())), ceilf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return ceilf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> floor(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(floorf(static_cast<float>(x.low())), floorf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return floorf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> sqrt(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(sqrtf(static_cast<float>(x.low())), sqrtf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return sqrtf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> pow(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(powf(static_cast<float>(x.low()), static_cast<float>(y.low())), powf(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return powf(x, y);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> mod(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(fmodf(static_cast<float>(x.low()), static_cast<float>(y.low())),
				fmodf(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmodf(x, y);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> exp(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> log(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return logf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> tanh(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(tanhf(static_cast<float>(x.low())), tanhf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanhf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> expm1(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x) - static_cast<__half2 >(vector_one<half>());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expm1f(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> log1p(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(vector_one<half>() + x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return log1pf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> sin(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2sin(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return sinf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> cos(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2cos(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return cosf(x);
#else
		return Vector<half>();
#endif
	}
	DEVICE_INLINE Vector<half> tan(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(tanf(static_cast<float>(x.low())), tanf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanf(x);
#else
		return Vector<half>();
#endif
	}

	DEVICE_INLINE half horizontal_add(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return x.low() + x.high();
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	DEVICE_INLINE half horizontal_mul(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return x.low() + x.high();
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	DEVICE_INLINE half horizontal_max(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return x.low() > x.high() ? x.low() : x.high();
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	DEVICE_INLINE half horizontal_min(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return x.low() < x.high() ? x.low() : x.high();
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	DEVICE_INLINE half horizontal_or(Vector<half> x)
	{
		return half(); // TODO
	}
	DEVICE_INLINE half horizontal_and(Vector<half> x)
	{
		return half(); // TODO
	}

} /* namespace numbers */

#endif /* CUDA_FP16_VECTOR_CUH_ */
