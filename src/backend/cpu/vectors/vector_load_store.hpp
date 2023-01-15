/*
 * vector_load_store.hpp
 *
 *  Created on: Jan 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_LOAD_STORE_HPP_
#define VECTORS_VECTOR_LOAD_STORE_HPP_

#include <cassert>
#include <x86intrin.h>

#include "vector_length.hpp"
#include "vector_macros.hpp"
#include "vector_utils.hpp"

namespace SIMD_NAMESPACE
{
#if SUPPORTS_SSE2
	static inline __m128d partial_load(const double *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_pd();
			case 1:
				return _mm_load_sd(ptr);
			case 2:
			default:
				return _mm_loadu_pd(ptr);
		}
	}
	static inline __m128 partial_load(const float *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_ps();
			case 1:
				return _mm_load_ss(ptr);
			case 2:
				return _mm_castsi128_ps(_mm_loadu_si64(ptr));
			case 3:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(ptr + 2);
				return _mm_movelh_ps(tmp1, tmp2);
			}
			case 4:
			default:
				return _mm_loadu_ps(ptr);
		}
	}
	static inline __m128i partial_load(const int64_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_loadu_si64(ptr);
			case 2:
			default:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline __m128i partial_load(const uint64_t *ptr, const int num) noexcept
	{
		return partial_load(reinterpret_cast<const int64_t*>(ptr), num);
	}
	static inline __m128i partial_load(const int32_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 2:
				return _mm_loadu_si64(ptr);
			case 3:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
			}
			case 4:
			default:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline __m128i partial_load(const uint32_t *ptr, const int num) noexcept
	{
		return partial_load(reinterpret_cast<const int32_t*>(ptr), num);
	}
	static inline __m128i partial_load(const int16_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_setr_epi16(ptr[0], 0u, 0u, 0u, 0u, 0u, 0u, 0u);
			case 2:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 3:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], 0u, 0u, 0u, 0u, 0u);
			case 4:
				return _mm_loadu_si64(ptr);
			case 5:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], 0u, 0u, 0u);
			case 6:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
			}
			case 7:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], 0u);
			case 8:
			default:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline __m128i partial_load(const uint16_t *ptr, const int num) noexcept
	{
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m128i partial_load(const int8_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				return _mm_setzero_si128();
			case 4:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 8:
				return _mm_loadu_si64(ptr);
			case 12:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
			}
			case 16:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
			default:
			{
				if (num > 16)
					return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
				else
				{
					int32_t tmp[4] = { 0, 0, 0, 0 };
					std::memcpy(tmp, ptr, num);
					return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
				}
			}
		}
	}
	static inline __m128i partial_load(const uint8_t *ptr, const int num) noexcept
	{
		return partial_load(reinterpret_cast<const int8_t*>(ptr), num);
	}

	static inline void partial_store(__m128d reg, double *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_store_sd(ptr, reg);
				break;
			case 2:
			default:
				_mm_storeu_pd(ptr, reg);
				break;
		}
	}
	static inline void partial_store(__m128 reg, float *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_store_ss(ptr, reg);
				break;
			case 2:
				_mm_storeu_si64(ptr, _mm_castps_si128(reg));
				break;
			case 3:
			{
				_mm_storeu_si64(ptr, _mm_castps_si128(reg));
				__m128 tmp = _mm_movehl_ps(reg, reg);
				_mm_store_ss(ptr + 2, tmp);
				break;
			}
			case 4:
			default:
				_mm_storeu_ps(ptr, reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, int64_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 2:
			default:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, uint64_t *ptr, const int num) noexcept
	{
		partial_store(reg, reinterpret_cast<int64_t*>(ptr), num);
	}
	static inline void partial_store(__m128i reg, int32_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(reg));
				break;
			case 2:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 3:
			{
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(reg), _mm_castsi128_ps(reg));
				_mm_store_ss(reinterpret_cast<float*>(ptr) + 2, tmp);
				break;
			}
			case 4:
			default:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, uint32_t *ptr, const int num) noexcept
	{
		partial_store(reg, reinterpret_cast<int32_t*>(ptr), num);
	}
	static inline void partial_store(__m128i reg, int16_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = _mm_extract_epi16(reg, 0);
				break;
			case 2:
				ptr[0] = _mm_extract_epi16(reg, 0);
				ptr[1] = _mm_extract_epi16(reg, 1);
				break;
			case 3:
			{
				ptr[0] = _mm_extract_epi16(reg, 0);
				ptr[1] = _mm_extract_epi16(reg, 1);
				ptr[2] = _mm_extract_epi16(reg, 2);
				break;
			}
			case 4:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 5:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				ptr[4] = _mm_extract_epi16(reg, 4);
				break;
			case 6:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				ptr[4] = _mm_extract_epi16(reg, 4);
				ptr[5] = _mm_extract_epi16(reg, 5);
				break;
			case 7:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				ptr[4] = _mm_extract_epi16(reg, 4);
				ptr[5] = _mm_extract_epi16(reg, 5);
				ptr[6] = _mm_extract_epi16(reg, 6);
				break;
			case 8:
			default:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, uint16_t *ptr, const int num) noexcept
	{
		partial_store(reg, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void partial_store(__m128i reg, int8_t *ptr, const int num) noexcept
	{
		assert(num >= 0);
		switch (num)
		{
			case 0:
				break;
			case 4:
				_mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(reg));
				break;
			case 8:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 12:
			{
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(reg), _mm_castsi128_ps(reg));
				_mm_store_ss(reinterpret_cast<float*>(ptr) + 2, tmp);
				break;
			}
			case 16:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			default:
			{
				if (num > 16)
					_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				else
				{
					int32_t tmp[4];
					_mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), reg);
					std::memcpy(ptr, tmp, num);
				}
				break;
			}
		}
	}
	static inline void partial_store(__m128i reg, uint8_t *ptr, const int num) noexcept
	{
		partial_store(reg, reinterpret_cast<int8_t*>(ptr), num);
	}
#endif

#if SUPPORTS_AVX

	static inline __m256d vector_load(const double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<double>())
			return _mm256_loadu_pd(ptr);
		else
		{
			if (num > vector_length<double>() / 2)
				return combine(_mm_loadu_pd(ptr), partial_load(ptr + vector_length<double>() / 2, num - vector_length<double>() / 2));
			else
				return combine(partial_load(ptr, num));
		}
	}
	static inline __m256 vector_load(const float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<float>())
			return _mm256_loadu_ps(ptr);
		else
		{
			if (num > vector_length<float>() / 2)
				return combine(_mm_loadu_ps(ptr), partial_load(ptr + vector_length<float>() / 2, num - vector_length<float>() / 2));
			else
				return combine(partial_load(ptr, num));
		}
	}
	static inline __m128i vector_load(const bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0); // TODO AVX512 adds full support for bfloat16 data
		return partial_load(reinterpret_cast<const uint16_t*>(ptr), num);
	}
	static inline __m128i vector_load(const float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0); // TODO AVX512 adds full support for float16 data
		return partial_load(reinterpret_cast<const uint16_t*>(ptr), num);
	}
	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline __m256i vector_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<T>())
			return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
		else
		{
			if (num > vector_length<T>() / 2)
				return combine(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)),
						partial_load(ptr + (vector_length<T>() / 2), num - vector_length<T>() / 2));
			else
				return combine(partial_load(ptr, sizeof(T) * num));
		}
	}

	static inline void vector_store(__m256d x, double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<double>())
			_mm256_storeu_pd(ptr, x);
		else
		{
			if (num > vector_length<double>() / 2)
			{
				_mm_storeu_pd(ptr, get_low(x));
				partial_store(get_high(x), ptr + vector_length<double>() / 2, num - vector_length<double>() / 2);
			}
			else
				partial_store(get_low(x), ptr, num);
		}
	}
	static inline void vector_store(__m256 x, float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<float>())
			_mm256_storeu_ps(ptr, x);
		else
		{
			if (num > vector_length<float>() / 2)
			{
				_mm_storeu_ps(ptr, get_low(x));
				partial_store(get_high(x), ptr + vector_length<float>() / 2, num - vector_length<float>() / 2);
			}
			else
				partial_store(get_low(x), ptr, num);
		}
	}
	static inline void vector_store(__m128i x, bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, reinterpret_cast<uint16_t*>(ptr), num);
	}
	static inline void vector_store(__m128i x, float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, reinterpret_cast<uint16_t*>(ptr), num);
	}
	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline void vector_store(__m256i x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		if (num == vector_length<T>())
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), x);
		else
		{
			if (num > vector_length<T>() / 2)
			{
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), get_low(x));
				partial_store(get_high(x), ptr + vector_length<T>() / 2, sizeof(T) * (num - vector_length<T>() / 2));
			}
			else
				partial_store(get_low(x), ptr, sizeof(T) * num);
		}
	}

#elif SUPPORTS_SSE2

	static inline __m128d vector_load(const double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		return partial_load(ptr, num);
	}
	static inline __m128 vector_load(const float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		return partial_load(ptr, num);
	}
	static inline __m128i vector_load(const bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0); // TODO AVX512 adds full support for bfloat16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m128i vector_load(const float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0); // TODO AVX512 adds full support for float16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline __m128i vector_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		return partial_load(ptr, sizeof(T) * num);
	}

	static inline void vector_store(__m128d x, double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, ptr, num);
	}
	static inline void vector_store(__m128 x, float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, ptr, num);
	}
	static inline void vector_store(__m128i x, bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void vector_store(__m128i x, float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline void vector_store(__m128i x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		partial_store(x, ptr, sizeof(T) * num);
	}

#else

	template<typename T>
	static inline T vector_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		return ptr[0];
	}
	template<typename T>
	static inline void vector_store(T x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0);
		ptr[0] = x;
	}
#endif

} /* SIMD_NAMESPACE */

#endif /* VECTORS_VECTOR_LOAD_STORE_HPP_ */
