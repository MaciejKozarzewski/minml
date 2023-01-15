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
#if (__CUDA_ARCH__ < BF16_COMPUTE_MIN_ARCH) or not HAS_BF16_HEADER
	DEVICE_INLINE __host__ __nv_bfloat16 float_to___nv_bfloat16(float x)
	{
		return reinterpret_cast<__nv_bfloat16*>(&x)[1];
	}
	DEVICE_INLINE __host__ float __nv_bfloat16_to_float(__nv_bfloat16 x) noexcept
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
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// __nv_bfloat16x2 m_data;
#else
			float m_data;
#endif
		public:
			__device__ Vector() // @suppress("Class members should be properly initialized")
			{
			}
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
//		__device__ Vector(__nv_bfloat162 x) :
//		m_data(x)
//		{
//		}
#else
			__device__ Vector(__nv_bfloat16 x) :
					m_data(__nv_bfloat16_to_float(x))
			{
			}
			__device__ Vector(float x) :
					m_data(x)
			{
			}
#endif
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
//		__device__ Vector(const __nv_bfloat16 *ptr, int num = 2)
//		{
//			load(ptr, num);
//		}
//		__device__ Vector(const float *ptr, int num = 2)
//		{
//			load(ptr, num);
//		}
//		__device__ void load(const __nv_bfloat16 *ptr, int num = 2)
//		{
//			assert(ptr != nullptr);
//			if (num >= 2)
//				m_data = __nv_bfloat162(ptr[0], ptr[1]);
//			else
//			{
//				if (num == 1)
//					m_data = __nv_bfloat162(ptr[0], 0.0f);
//				else
//					m_data = __nv_bfloat162(0.0f, 0.0f);
//			}
//		}
//		__device__ void load(const float *ptr, int num = 2)
//		{
//			assert(ptr != nullptr);
//			if (num >= 2)
//				m_data = __nv_bfloat162(ptr[0], ptr[1]);
//			else
//			{
//				if (num == 1)
//					m_data = __nv_bfloat162(ptr[0], 0.0f);
//				else
//					m_data = __nv_bfloat162(0.0f, 0.0f);
//			}
//		}
//		__device__ void store(__nv_bfloat16 *ptr, int num = 2) const
//		{
//			assert(ptr != nullptr);
//			switch (num)
//			{
//				default:
//				case 2:
//					ptr[0] = m_data.x;
//					ptr[1] = m_data.y;
//					break;
//				case 1:
//					ptr[0] = m_data.x;
//					break;
//			}
//		}
//		__device__ void store(float *ptr, int num = 2) const
//		{
//			assert(ptr != nullptr);
//			switch (num)
//			{
//				default:
//				case 2:
//					ptr[0] = static_cast<float>(m_data.x);
//					ptr[1] = static_cast<float>(m_data.y);
//					break;
//				case 1:
//					ptr[0] = m_data.x;
//					break;
//			}
//		}
//		__device__ operator __nv_bfloat162() const
//		{
//			return m_data;
//		}
//		__device__ Vector<__nv_bfloat16> operator-() const
//		{
//			return Vector<__nv_bfloat16>(-m_data);
//		}
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
					m_data = __nv_bfloat16_to_float(ptr[0]);
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
					ptr[0] = float_to___nv_bfloat16(m_data);
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
			__device__  __nv_bfloat16 get() const
			{
				return float_to___nv_bfloat16(m_data);
			}
#endif
			__device__ Vector<__nv_bfloat16> operator~() const
			{
				const int32_t tmp = ~reinterpret_cast<const int32_t*>(&m_data)[0];
				return Vector<__nv_bfloat16 >(reinterpret_cast<const __nv_bfloat16*>(&tmp)[0]);
			}
	};

	template<>
	DEVICE_INLINE constexpr int vector_length<__nv_bfloat16 >()
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
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
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(static_cast<__nv_bfloat16x2>(lhs) + static_cast<__nv_bfloat16x2>(rhs));
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) + static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator-(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(static_cast<__nv_bfloat16x2>(lhs) - static_cast<__nv_bfloat16x2>(rhs));
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) - static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator*(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(static_cast<__nv_bfloat16x2>(lhs) * static_cast<__nv_bfloat16x2>(rhs));
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) * static_cast<float>(rhs));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> operator/(const Vector<__nv_bfloat16> &lhs, const Vector<__nv_bfloat16> &rhs)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(static_cast<__nv_bfloat16x2>(lhs) / static_cast<__nv_bfloat16x2>(rhs));
#else
		return Vector<__nv_bfloat16 >(static_cast<float>(lhs) / static_cast<float>(rhs));
#endif
	}

	DEVICE_INLINE Vector<__nv_bfloat16> sgn(Vector<__nv_bfloat16> x) noexcept
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		__nv_bfloat16x2 tmp = x;
		__nv_bfloat16x2 result;
		result.x = static_cast<__nv_bfloat16>(internal::sgn(static_cast<float>(tmp.x)));
		result.y = static_cast<__nv_bfloat16>(internal::sgn(static_cast<float>(tmp.y)));
		return result;
#else
		return Vector<__nv_bfloat16 >(internal::sgn(static_cast<float>(x)));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> abs(Vector<__nv_bfloat16> x) noexcept
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		__nv_bfloat16x2 tmp = x;
		__nv_bfloat16x2 result;
		result.x = static_cast<__nv_bfloat16>(fabsf(static_cast<float>(tmp.x)));
		result.y = static_cast<__nv_bfloat16>(fabsf(static_cast<float>(tmp.y)));
		return result;
#else
		return Vector<__nv_bfloat16 >(fabsf(static_cast<float>(x)));
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> max(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(fmax(static_cast<float>(x.low()), static_cast<float>(y.low())), fmax(static_cast<float>(x.high()), static_cast<float>(y.high())));
#else
		return fmax(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> min(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(fmin(static_cast<float>(x.low()), static_cast<float>(y.low())), fmin(static_cast<float>(x.high()), static_cast<float>(y.high())));
#else
		return fmin(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> ceil(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(ceilf(static_cast<float>(x.low())), ceilf(static_cast<float>(x.high())));
#else
		return ceilf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> floor(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(floorf(static_cast<float>(x.low())), floorf(static_cast<float>(x.high())));
#else
		return floorf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> sqrt(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return Vector<__nv_bfloat16>(sqrtf(static_cast<float>(x.low())), sqrtf(static_cast<float>(x.high())));
#else
		return sqrtf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> pow(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return powf(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> mod(Vector<__nv_bfloat16> x, Vector<__nv_bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return fmodf(x, y);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> exp(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return expf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> log(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return logf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> tanh(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return tanhf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> expm1(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return expm1f(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> log1p(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return log1pf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> sin(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return sinf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> cos(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return cosf(x);
#endif
	}
	DEVICE_INLINE Vector<__nv_bfloat16> tan(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return tanf(x);
#endif
	}

	DEVICE_INLINE __nv_bfloat16 horizontal_add(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_mul(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_max(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return x.get();
#endif
	}
	DEVICE_INLINE __nv_bfloat16 horizontal_min(Vector<__nv_bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
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
