/*
 * vector_store.hpp
 *
 *  Created on: Mar 30, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_STORE_HPP_
#define VECTORS_VECTOR_STORE_HPP_

#include "vector_utils.hpp"
#include "register_type.hpp"

namespace SIMD_NAMESPACE
{
	template<RegisterType RT>
	struct Storer;

	template<>
	struct Storer<SCALAR>
	{
			template<typename T>
			void operator()(T *dst, T value, int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 1);
				if (num == 1)
					dst[0] = value;
			}
	};

#if SUPPORTS_SSE2
	template<>
	struct Storer<XMM>
	{
			void operator()(double *dst, __m128d value, const int num) const noexcept
			{
				switch (num)
				{
					case 0:
						break;
					case 1:
						_mm_store_sd(dst, value);
						break;
					case 2:
					default:
						_mm_storeu_pd(dst, value);
						break;
				}
			}
			void operator()(float *dst, __m128 value, const int num) const noexcept
			{
				switch (num)
				{
					case 0:
						break;
					case 1:
						_mm_store_ss(dst, value);
						break;
					case 2:
						_mm_storeu_si64(dst, _mm_castps_si128(value));
						break;
					case 3:
					{
						_mm_storeu_si64(dst, _mm_castps_si128(value));
						__m128 tmp = _mm_movehl_ps(value, value);
						_mm_store_ss(dst + 2, tmp);
						break;
					}
					case 4:
					default:
						_mm_storeu_ps(dst, value);
						break;
				}
			}
			void operator()(int64_t *dst, __m128i value, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 2);
				switch (num)
				{
					case 0:
						break;
					case 1:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						break;
					case 2:
					default:
						_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
						break;
				}
			}
			void operator()(uint64_t *dst, __m128i value, const int num) const noexcept
			{
				return operator()(reinterpret_cast<int64_t*>(dst), value, num);
			}
			void operator()(int32_t *dst, __m128i value, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 4);
				switch (num)
				{
					case 0:
						break;
					case 1:
						_mm_store_ss(reinterpret_cast<float*>(dst), _mm_castsi128_ps(value));
						break;
					case 2:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						break;
					case 3:
					{
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(value), _mm_castsi128_ps(value));
						_mm_store_ss(reinterpret_cast<float*>(dst) + 2, tmp);
						break;
					}
					case 4:
					default:
						_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
						break;
				}
			}
			void operator()(uint32_t *dst, __m128i value, const int num) const noexcept
			{
				return operator()(reinterpret_cast<int32_t*>(dst), value, num);
			}
			void operator()(int16_t *dst, __m128i value, const int num) const noexcept
			{
				assert(src != nullptr);
				assert(0 <= num && num <= 8);
				switch (num)
				{
					case 0:
						break;
					case 1:
						dst[0] = _mm_extract_epi16(value, 0);
						break;
					case 2:
						dst[0] = _mm_extract_epi16(value, 0);
						dst[1] = _mm_extract_epi16(value, 1);
						break;
					case 3:
					{
						dst[0] = _mm_extract_epi16(value, 0);
						dst[1] = _mm_extract_epi16(value, 1);
						dst[2] = _mm_extract_epi16(value, 2);
						break;
					}
					case 4:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						break;
					case 5:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						dst[4] = _mm_extract_epi16(value, 4);
						break;
					case 6:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						dst[4] = _mm_extract_epi16(value, 4);
						dst[5] = _mm_extract_epi16(value, 5);
						break;
					case 7:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						dst[4] = _mm_extract_epi16(value, 4);
						dst[5] = _mm_extract_epi16(value, 5);
						dst[6] = _mm_extract_epi16(value, 6);
						break;
					case 8:
					default:
						_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
						break;
				}
			}
			void operator()(uint16_t *dst, __m128i value, const int num) const noexcept
			{
				return operator()(reinterpret_cast<int16_t*>(dst), value, num);
			}
			void operator()(int8_t *dst, __m128i value, const int num) const noexcept
			{
				assert(dst != nullptr);
				assert(0 <= num && num <= 16);
				switch (num)
				{
					case 0:
						break;
					case 4:
						_mm_store_ss(reinterpret_cast<float*>(dst), _mm_castsi128_ps(value));
						break;
					case 8:
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						break;
					case 12:
					{
						_mm_storeu_si64(reinterpret_cast<__m128i*>(dst), value);
						__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(value), _mm_castsi128_ps(value));
						_mm_store_ss(reinterpret_cast<float*>(dst) + 2, tmp);
						break;
					}
					case 16:
						_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
						break;
					default:
					{
						int32_t tmp[4];
						_mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), value);
						std::memcpy(dst, tmp, num);
						break;
					}
				}
			}
			void operator()(uint8_t *dst, __m128i value, const int num) const noexcept
			{
				operator()(reinterpret_cast<int8_t*>(dst), value, num);
			}
	};
#endif

#if SUPPORTS_AVX
	template<>
	struct Storer<YMM>
	{
			void operator()(double *dst, __m256d value, int num) const noexcept
			{
				assert(dst != nullptr);
				constexpr int length = vector_size<double, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					_mm256_storeu_pd(dst, value);
				else
				{
					Storer<XMM> store;
					if (num > length / 2)
					{
						_mm_storeu_pd(dst, get_low(value));
						store(dst + length / 2, get_high(value), num - length / 2);
					}
					else
						store(dst, get_low(value), num);
				}
			}
			void operator()(float *dst, __m256 value, int num) const noexcept
			{
				assert(dst != nullptr);
				constexpr int length = vector_size<float, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					_mm256_storeu_ps(dst, value);
				else
				{
					Storer<XMM> store;
					if (num > length / 2)
					{
						_mm_storeu_ps(dst, get_low(value));
						store(dst + length / 2, get_high(value), num - length / 2);
					}
					else
						store(dst, get_low(value), num);
				}
			}
			template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
			void operator()(T *dst, __m256i value, int num) const noexcept
			{
				assert(dst != nullptr);
				constexpr int length = vector_size<T, YMM>();
				assert(0 <= num && num <= length);
				if (num == length)
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), value);
				else
				{
					Storer<XMM> store;
					if (num > length / 2)
					{
						_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), get_low(value));
						store(dst + length / 2, get_high(value), sizeof(T) * (num - length / 2));
					}
					else
						store(dst, get_low(value), sizeof(T) * num);
				}
			}
	};
#endif

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_VECTOR_STORE_HPP_ */
