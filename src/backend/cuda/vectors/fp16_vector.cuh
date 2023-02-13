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
			HOST_DEVICE Vector() // @suppress("Class members should be properly initialized")
			{
			}
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
			HOST_DEVICE Vector(half2 x) :
					m_data(x)
			{
			}
			HOST_DEVICE Vector(half x) :
					m_data(x, x)
			{
			}
			HOST_DEVICE Vector(half x, half y) :
					m_data(x, y)
			{
			}
			HOST_DEVICE Vector(float x) :
					m_data(x, x)
			{
			}
			HOST_DEVICE Vector(float x, float y) :
					m_data(x, y)
			{
			}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
			HOST_DEVICE Vector(half x) :
					m_data(x)
			{
			}
			HOST_DEVICE Vector(float x) :
					m_data(x)
			{
			}
#else
			HOST_DEVICE Vector(half x)
			{
				assert(false);
			}
			HOST_DEVICE Vector(float x)
			{
				assert(false);
			}
			HOST_DEVICE Vector(double x)
			{
				assert(false);
			}
#endif
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
			HOST_DEVICE Vector(const half * __restrict__ ptr, int num = 2)
			{
				load(ptr, num);
			}
			HOST_DEVICE Vector(const float * __restrict__ ptr, int num = 2)
			{
				load(ptr, num);
			}
			HOST_DEVICE void load(const half * __restrict__ ptr, int num = 2)
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
			HOST_DEVICE void load(const float * __restrict__ ptr, int num = 2)
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
			HOST_DEVICE void store(half * __restrict__ ptr, int num = 2) const
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
			HOST_DEVICE void store(float * __restrict__ ptr, int num = 2) const
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
			HOST_DEVICE operator half2() const
			{
				return m_data;
			}
			HOST_DEVICE half low() const
			{
				return m_data.x;
			}
			HOST_DEVICE half high() const
			{
				return m_data.y;
			}
			HOST_DEVICE Vector<half> operator-() const
			{
				return __hneg2(m_data);
			}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
			HOST_DEVICE Vector(const half * __restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			HOST_DEVICE Vector(const float * __restrict__ ptr, int num = 1)
			{
				load(ptr, num);
			}
			HOST_DEVICE void load(const half * __restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = static_cast<float>(ptr[0]);
			}
			HOST_DEVICE void load(const float * __restrict__ ptr, int num = 1)
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					m_data = ptr[0];
			}
			HOST_DEVICE void store(half * __restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			HOST_DEVICE void store(float * __restrict__ ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				assert(num >= 0);
				if (num >= 1)
					ptr[0] = m_data;
			}
			HOST_DEVICE operator float() const
			{
				return m_data;
			}
			HOST_DEVICE Vector<half> operator-() const
			{
				return Vector<half>(-m_data);
			}
			HOST_DEVICE half get() const
			{
				return m_data;
			}
#else
			HOST_DEVICE Vector(const half *__restrict__ ptr, int num = 0)
			{
				assert(false);
			}
			HOST_DEVICE Vector(const float *__restrict__ ptr, int num = 0)
			{
				assert(false);
			}
			HOST_DEVICE void load(const half *__restrict__ ptr, int num = 0)
			{
				assert(false);
			}
			HOST_DEVICE void load(const float *__restrict__ ptr, int num = 0)
			{
				assert(false);
			}
			HOST_DEVICE void store(half *__restrict__ ptr, int num = 0) const
			{
				assert(false);
			}
			HOST_DEVICE void store(float *__restrict__ ptr, int num = 0) const
			{
				assert(false);
			}
			HOST_DEVICE operator float() const
			{
				assert(false);
				return 0.0f;
			}
			HOST_DEVICE Vector<half> operator-() const
			{
				assert(false);
				return Vector<half>(0.0f);
			}
#endif
			HOST_DEVICE Vector<half> operator~() const
			{
#if __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
				const uint32_t tmp = ~reinterpret_cast<const uint32_t*>(&m_data)[0];
				return Vector<half>(reinterpret_cast<const half*>(&tmp)[0]);
#else
				assert(false);
				return Vector<half>();
#endif
			}
	};

	template<>
	HOST_DEVICE_INLINE constexpr int vector_length<half>()
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
	HOST_DEVICE_INLINE Vector<half> vector_zero()
	{
		return Vector<half>(0.0f);
	}
	template<>
	HOST_DEVICE_INLINE Vector<half> vector_one()
	{
		return Vector<half>(1.0f);
	}
	template<>
	HOST_DEVICE_INLINE Vector<half> vector_epsilon()
	{
		return Vector<half>(0.00006103515625f);
	}

	HOST_DEVICE_INLINE Vector<half> operator+(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hadd2(lhs, rhs);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) + static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> operator-(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hsub2(lhs, rhs);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) - static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> operator*(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hmul2(lhs, rhs);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) * static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> operator/(const Vector<half> &lhs, const Vector<half> &rhs)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __h2div(lhs, rhs);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(static_cast<float>(lhs) / static_cast<float>(rhs));
#else
		return Vector<half>();
#endif
	}

	// TODO add fused multiply-add instructions (__hfma2)

	HOST_DEVICE_INLINE Vector<half> sgn(Vector<half> x) noexcept
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
	HOST_DEVICE_INLINE Vector<half> abs(Vector<half> x) noexcept
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __habs2(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Vector<half>(fabsf(static_cast<float>(x)));
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> max(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hmax2(x, y);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmax(x, y);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> min(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hmin2(x, y);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmin(x, y);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> ceil(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2ceil(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return ceilf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> floor(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2floor(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return floorf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> sqrt(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2sqrt(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return sqrtf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> pow(Vector<half> x, Vector<half> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(powf(static_cast<float>(x.low()), static_cast<float>(y.low())), powf(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return powf(x, y);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> mod(Vector<half> x, Vector<half> y)
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
	HOST_DEVICE_INLINE Vector<half> exp(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> log(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return logf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> tanh(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(tanhf(static_cast<float>(x.low())), tanhf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanhf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> expm1(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x) - static_cast<__half2 >(vector_one<half>());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expm1f(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> log1p(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(vector_one<half>() + x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return log1pf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> sin(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2sin(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return sinf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> cos(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2cos(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return cosf(x);
#else
		return Vector<half>();
#endif
	}
	HOST_DEVICE_INLINE Vector<half> tan(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Vector<half>(tanf(static_cast<float>(x.low())), tanf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanf(x);
#else
		return Vector<half>();
#endif
	}

	HOST_DEVICE_INLINE half horizontal_add(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		__hadd(x.low(), x.high());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	HOST_DEVICE_INLINE half horizontal_mul(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		__hmul(x.low(), x.high());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	HOST_DEVICE_INLINE half horizontal_max(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hmax(x.low(), x.high());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	HOST_DEVICE_INLINE half horizontal_min(Vector<half> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return __hmin(x.low(), x.high());
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return x.get();
#else
		return half();
#endif
	}
	HOST_DEVICE_INLINE half horizontal_or(Vector<half> x)
	{
		return half(); // TODO
	}
	HOST_DEVICE_INLINE half horizontal_and(Vector<half> x)
	{
		return half(); // TODO
	}

} /* namespace numbers */

#endif /* CUDA_FP16_VECTOR_CUH_ */
