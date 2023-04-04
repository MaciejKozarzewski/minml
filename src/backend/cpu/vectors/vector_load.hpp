/*
 * vector_load.hpp
 *
 *  Created on: Mar 30, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_LOAD_HPP_
#define VECTORS_VECTOR_LOAD_HPP_

#include "vector_utils.hpp"
#include "register_type.hpp"

namespace SIMD_NAMESPACE
{
	template<RegisterType RT>
	struct Loader;

	template<>
	struct Loader<SCALAR>
	{
			template<typename T>
			T operator()(const T *src, int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 1);
				return (num == 0) ? T { } : src[0];
			}
	};

#if SUPPORTS_SSE2
	template<>
	struct Loader<XMM>
	{
			__m128d operator()(const double *src, const int num) const noexcept
			{
				switch (num)
				{
					case 0:
						return _mm_setzero_pd();
					case 1:
						return _mm_load_sd(src);
					case 2:
					default:
						return _mm_loadu_pd(src);
				}
			}
			__m128 operator()(const float *src, const int num) const noexcept
			{
				switch (num)
				{
					case 0:
						return _mm_setzero_ps();
					case 1:
						return _mm_load_ss(src);
					case 2:
						return _mm_castsi128_ps(_mm_loadu_si64(src));
					case 3:
					{
						__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(src));
						__m128 tmp2 = _mm_load_ss(src + 2);
						return _mm_movelh_ps(tmp1, tmp2);
					}
					case 4:
					default:
						return _mm_loadu_ps(src);
				}
			}
			__m128i operator()(const int64_t *src, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 2);
				switch (num)
				{
					case 0:
						return _mm_setzero_si128();
					case 1:
						return _mm_loadu_si64(src);
					case 2:
					default:
						return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
				}
			}
			__m128i operator()(const uint64_t *src, const int num) const noexcept
			{
				return operator()(reinterpret_cast<const int64_t*>(src), num);
			}
			__m128i operator()(const int32_t *src, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 4);
				switch (num)
				{
					case 0:
						return _mm_setzero_si128();
					case 1:
						return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(src)));
					case 2:
						return _mm_loadu_si64(src);
					case 3:
					{
						__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(src));
						__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(src) + 2);
						return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
					}
					case 4:
					default:
						return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
				}
			}
			__m128i operator()(const uint32_t *src, const int num) const noexcept
			{
				return operator()(reinterpret_cast<const int32_t*>(src), num);
			}
			__m128i operator()(const int16_t *src, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 8);
				switch (num)
				{
					case 0:
						return _mm_setzero_si128();
					case 1:
						return _mm_setr_epi16(src[0], 0u, 0u, 0u, 0u, 0u, 0u, 0u);
					case 2:
						return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(src)));
					case 3:
						return _mm_setr_epi16(src[0], src[1], src[2], 0u, 0u, 0u, 0u, 0u);
					case 4:
						return _mm_loadu_si64(src);
					case 5:
						return _mm_setr_epi16(src[0], src[1], src[2], src[3], src[4], 0u, 0u, 0u);
					case 6:
					{
						__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(src));
						__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(src) + 2);
						return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
					}
					case 7:
						return _mm_setr_epi16(src[0], src[1], src[2], src[3], src[4], src[5], src[6], 0u);
					case 8:
					default:
						return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
				}
			}
			__m128i operator()(const uint16_t *src, const int num) const noexcept
			{
				return operator()(reinterpret_cast<const int16_t*>(src), num);
			}
			__m128i operator()(const int8_t *src, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 16);
				switch (num)
				{
					case 0:
						return _mm_setzero_si128();
					case 4:
						return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(src)));
					case 8:
						return _mm_loadu_si64(src);
					case 12:
					{
						__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(src));
						__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(src) + 2);
						return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
					}
					case 16:
						return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
					default:
					{
						int32_t tmp[4] = { 0, 0, 0, 0 };
						std::memcpy(tmp, src, num);
						return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
					}
				}
			}
			__m128i operator()(const uint8_t *src, const int num) const noexcept
			{
				return operator()(reinterpret_cast<const int8_t*>(src), num);
			}
	};
#endif

#if SUPPORTS_AVX
	template<>
	struct Loader<YMM>
	{
			__m256d operator()(const double *src, int num) const noexcept
			{
				assert(src != nullptr);
				constexpr int length = vector_size<double, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					return _mm256_loadu_pd(src);
				else
				{
					Loader<XMM> load;
					if (num > length / 2)
						return combine(_mm_loadu_pd(src), load(src + length / 2, num - length / 2));
					else
						return combine(load(src, num), _mm_setzero_pd());
				}
			}
			__m256 operator()(const float *src, int num) const noexcept
			{
				assert(src != nullptr);
				constexpr int length = vector_size<float, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					return _mm256_loadu_ps(src);
				else
				{
					Loader<XMM> load;
					if (num > length / 2)
						return combine(_mm_loadu_ps(src), load(src + length / 2, num - length / 2));
					else
						return combine(load(src, num), _mm_setzero_ps());
				}
			}
			template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
			__m256i operator()(const T *src, int num) const noexcept
			{
				assert(src != nullptr);
				constexpr int length = vector_size<T, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
				else
				{
					Loader<XMM> load;
					if (num > length / 2)
						return combine(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)), load(src + length / 2, num - length / 2));
					else
						return combine(load(src, sizeof(T) * num), _mm_setzero_si128());
				}
			}
	};
#endif

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_VECTOR_LOAD_HPP_ */
