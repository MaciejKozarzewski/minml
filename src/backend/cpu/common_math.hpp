/*
 * common_math.hpp
 *
 *  Created on: Oct 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_COMMON_MATH_HPP_
#define BACKEND_CPU_COMMON_MATH_HPP_

#include <cmath>
#include <cassert>

namespace ml
{
	namespace cpu
	{

		template<typename T>
		T round_small_to_zero(T x) noexcept
		{
			return (fabsf(x) < 1.0e-7f) ? static_cast<T>(0) : x;
		}
		template<typename T>
		T safe_log(T x) noexcept
		{
			return std::log(1.0e-8f + x);
		}
		template<typename T>
		T cross_entropy(T output, T target) noexcept
		{
			return -target * safe_log(output) - (static_cast<T>(1) - target) * safe_log(static_cast<T>(1) - output);
		}
		template<typename T>
		T square(T x) noexcept
		{
			return x * x;
		}

		template<typename T>
		T sigmoid(T x) noexcept
		{
			return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
		}
		template<typename T>
		T relu(T x) noexcept
		{
			return std::max(static_cast<T>(0), x);
		}
		template<typename T>
		T leaky_relu(T x) noexcept
		{
			return (x > static_cast<T>(0)) ? x : static_cast<T>(0.1) * x;
		}
		template<typename T>
		T approx_gelu(T x) noexcept
		{
			return x / (static_cast<T>(1) + std::exp(-static_cast<T>(1.6849) * x));
		}
	}
}

#endif /* BACKEND_CPU_COMMON_MATH_HPP_ */
