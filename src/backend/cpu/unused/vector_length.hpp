/*
 * vector_length.hpp
 *
 *  Created on: Jan 6, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_SIMD_LENGTH_HPP_
#define VECTORS_SIMD_LENGTH_HPP_

#include "vector_macros.hpp"
#include "vector_utils.hpp"

namespace SIMD_NAMESPACE
{
#if SUPPORTS_AVX
	static inline constexpr int register_size() noexcept
	{
		return 256 / 8;
	}
#elif SUPPORTS_SSE2
	static inline constexpr int register_size() noexcept
	{
		return 128 / 8;
	}
#endif

#if SUPPORTS_AVX
	template<typename T>
	inline constexpr int vector_length() noexcept
	{
		return register_size() / sizeof(T);
	}
#elif SUPPORTS_SSE2
	template<typename T>
	inline constexpr int vector_length() noexcept
	{
		return register_size() / sizeof(T);
	}
	template<>
	inline constexpr int vector_length<float16>() noexcept
	{
		return register_size() / sizeof(float);
	}
	template<>
	inline constexpr int vector_length<bfloat16>() noexcept
	{
		return register_size() / sizeof(float);
	}
#else
	template<typename T>
	static inline constexpr int vector_length() noexcept
	{
		return 1;
	}
#endif

} /* SIMD_NAMESPACE */

#endif /* VECTORS_SIMD_LENGTH_HPP_ */
