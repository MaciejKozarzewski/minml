/*
 * bf16_vector.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef BF16_NUMBER_CUH_
#define BF16_NUMBER_CUH_

#include "generic_vector.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cassert>

namespace vectors
{
#if __CUDA_ARCH__ < BF16_COMPUTE_MIN_ARCH
	DEVICE_INLINE __host__ __nv_bfloat16 float_to_bfloat16(float x)
	{
		return reinterpret_cast<__nv_bfloat16*>(&x)[1];
	}
	DEVICE_INLINE __host__ float bfloat16_to_float(__nv_bfloat16 x) noexcept
	{
		float result = 0.0f;
		reinterpret_cast<__nv_bfloat16*>(&result)[1] = x;
		return result;
	}
#endif

	template<>
	class Vector<__nv_bfloat16>
	{
		private:
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
			__nv_bfloat162 m_data;
#else
			float m_data;
#endif
		public:
			__device__ Vector() // @suppress("Class members should be properly initialized")
			{
			}
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
			__device__ Vector(__nv_bfloat162 x) :
					m_data(x)
			{
			}
			__device__ Vector(__nv_bfloat16 x) :
					m_data(x, x)
			{
			}
			__device__ Vector(__nv_bfloat16 x, __nv_bfloat16 y) :
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
#else
			__device__ Vector(__nv_bfloat16 x) :
					m_data(bfloat16_to_float(x))
			{
			}
			__device__ Vector(float x) :
					m_data(x)
			{
			}
#endif
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
			__device__ Vector(const __nv_bfloat16 *ptr, int num = 2)
			{
				load(ptr, num);
			}
			__device__ Vector(const float *ptr, int num = 2)
			{
				load(ptr, num);
			}
			__device__ void load(const __nv_bfloat16 *ptr, int num = 2)
			{
				assert(ptr != nullptr);
				if (num >= 2)
				{
					if (is_aligned<__nv_bfloat162>(ptr))
						m_data = reinterpret_cast<const __nv_bfloat162*>(ptr)[0];
					else
						m_data = __nv_bfloat162(ptr[0], ptr[1]);
				}
				else
				{
					if (num == 1)
						m_data = __nv_bfloat162(ptr[0], 0.0f);
					else
						m_data = __nv_bfloat162(0.0f, 0.0f);
				}
			}
			__device__ void load(const float *ptr, int num = 2)
			{
				assert(ptr != nullptr);
				if (num >= 2)
					m_data = __nv_bfloat162(ptr[0], ptr[1]);
				else
				{
					if (num == 1)
						m_data = __nv_bfloat162(ptr[0], 0.0f);
					else
						m_data = __nv_bfloat162(0.0f, 0.0f);
				}
			}
			__device__ void store(__nv_bfloat16 *ptr, int num = 2) const
			{
				assert(ptr != nullptr);
				if (num >= 2)
				{
					if (is_aligned<__nv_bfloat162>(ptr))
						reinterpret_cast<__nv_bfloat162*>(ptr)[0] = m_data;
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
			__device__ void store(float *ptr, int num = 2) const
			{
				assert(ptr != nullptr);
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
			__device__ operator __nv_bfloat162() const
			{
				return m_data;
			}
			__device__ __nv_bfloat16 low() const
			{
				return m_data.x;
			}
			__device__ __nv_bfloat16 high() const
			{
				return m_data.y;
			}
			__device__ Vector<__nv_bfloat16> operator-() const
			{
				return __hneg2(m_data);
			}
#else
			__device__ Vector(const __nv_bfloat16 *ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ Vector(const float *ptr, int num = 1)
			{
				load(ptr, num);
			}
			__device__ void load(const __nv_bfloat16 *ptr, int num = 1)
			{
				assert(ptr != nullptr);
				if (num >= 1)
					m_data = bfloat16_to_float(ptr[0]);
			}
			__device__ void load(const float *ptr, int num = 1)
			{
				assert(ptr != nullptr);
				if (num >= 1)
					m_data = ptr[0];
			}
			__device__ void store(__nv_bfloat16 *ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				if (num >= 1)
					ptr[0] = float_to_bfloat16(m_data);
			}
			__device__ void store(float *ptr, int num = 1) const
			{
				assert(ptr != nullptr);
				if (num >= 1)
					ptr[0] = m_data;
			}
			__device__ operator float() const
			{
				return m_data;
			}
			__device__ Vector<__nv_bfloat16> operator-() const
			{
				return Vector<__nv_bfloat16 >(-m_data);
			}
			__device__    __nv_bfloat16 get() const
			{
				return float_to_bfloat16(m_data);
			}
#endif
			__device__ Vector<__nv_bfloat16> operator~() const
			{
				const uint32_t tmp = ~reinterpret_cast<const uint32_t*>(&m_data)[0];
				return Vector<__nv_bfloat16 >(reinterpret_cast<const __nv_bfloat16*>(&tmp)[0]);
			}
	};

	template<>
	DEVICE_INLINE constexpr int vector_length<__nv_bfloat16 >()
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return 2;
#else
		return 1;
#endif
	}

	template<>
	DEVICE_INLINE Vector<__nv_bfloat16> vector_zero()
	{
		return Vector<__nv_bfloat16 >(0.0f);
	}
	template<>
	DEVICE_INLINE Vector<__nv_bfloat16> vector_one()
	{
		return Vector<__nv_bfloat16 >(1.0f);
	}
	template<>
	DEVICE_INLINE Vector<__nv_bfloat16> vector_epsilon()
	{
		return Vector<__nv_bfloat16 >(1.1920928955078125e-7f);
	}

	DEVICE_INLINE Vector<__nv_bfloat16> operator+(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hadd2(lhs, rhs);
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) + static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator-(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hsub2(lhs, rhs);
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) - static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator*(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmul2(lhs, rhs);
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) * static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator/(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __h2div(lhs, rhs);
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) / static_cast<float>(rhs));
#endif
	}

	// TODO add fused multiply-add operations (__hfma2 or __hfma2_relu)
	// TODO add complex variants __hcmadd
	// TODO add comparison operators (they are supported by NVIDIA, for example __heq2_mask in 'cuda_bf16.h')

	DEVICE_INLINE Vector<__nv_bfloat16> sgn(Vector<__nv_bfloat16> x) noexcept
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		__nv_bfloat162 result;
		result.x = static_cast<__nv_bfloat16>(internal::sgn(static_cast<float>(x.low())));
		result.y = static_cast<__nv_bfloat16>(internal::sgn(static_cast<float>(x.high())));
		return result;
#else
		return Vector<__nv_bfloat16 >(internal::sgn(static_cast<float>(x)));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> abs(Vector<__nv_bfloat16> x) noexcept
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __habs2(x);
#else
		return Vector<__nv_bfloat16 >(fabsf(static_cast<float>(x)));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> max(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmax2(x, y);
#else
		return fmax(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> min(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmin2(x, y);
#else
		return fmin(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> ceil(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2ceil(x);
#else
		return ceilf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> floor(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2floor(x);
#else
		return floorf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> sqrt(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2sqrt(x);
#else
		return sqrtf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> pow(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		__nv_bfloat162 result;
		result.x = static_cast<__nv_bfloat16>(powf(static_cast<float>(x.low()), static_cast<float>(y.low())));
		result.y = static_cast<__nv_bfloat16>(powf(static_cast<float>(x.high()), static_cast<float>(y.high())));
		return result;
#else
		return powf(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> mod(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		__nv_bfloat162 result;
		result.x = static_cast<__nv_bfloat16>(fmodf(static_cast<float>(x.low()), static_cast<float>(y.low())));
		result.y = static_cast<__nv_bfloat16>(fmodf(static_cast<float>(x.high()), static_cast<float>(y.high())));
		return result;
#else
		return fmodf(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> exp(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2exp(x);
#else
		return expf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> log(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2log(x);
#else
		return logf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> tanh(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return Vector<__nv_bfloat16>(tanhf(static_cast<float>(x.low())), tanhf(static_cast<float>(x.high())));
#else
		return tanhf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> expm1(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return Vector<__nv_bfloat16>(expm1f(static_cast<float>(x.low())), expm1f(static_cast<float>(x.high())));
#else
		return expm1f(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> log1p(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return Vector<__nv_bfloat16>(log1pf(static_cast<float>(x.low())), log1pf(static_cast<float>(x.high())));
#else
		return log1pf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> sin(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2sin(x);
#else
		return sinf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> cos(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return h2cos(x);
#else
		return cosf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> tan(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return Vector<__nv_bfloat16>(tanf(static_cast<float>(x.low())), tanf(static_cast<float>(x.high())));
#else
		return tanf(x);
#endif
	}

	DEVICE_INLINE __nv_bfloat16 horizontal_add(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hadd(x.low(), x.high());
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_mul(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmul(x.low(), x.high());
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_max(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmax(x.low(), x.high());
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_min(Vector<__nv_bfloat16> x)
	{
#if __CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH
		return __hmin(x.low(), x.high());
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_or(Vector<__nv_bfloat16> x)
	{
		return __nv_bfloat16(); // TODO
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_and(Vector<__nv_bfloat16> x)
	{
		return __nv_bfloat16(); // TODO
	}

} /* namespace numbers */

#endif /* BF16_NUMBER_CUH_ */
