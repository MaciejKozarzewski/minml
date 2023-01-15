/*
 * simd_functions.hpp
 *
 *  Created on: Dec 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_FUNCTIONS_HPP_
#define VECTORS_VECTOR_FUNCTIONS_HPP_

#include "bf16_vector.hpp"
#include "fp16_vector.hpp"
#include "fp32_vector.hpp"
#include "generic_vector.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T>
	static inline Vector<T> square(Vector<T> x) noexcept
	{
		return x * x;
	}

	template<typename T>
	static inline Vector<T> pow(Vector<T> x, Vector<T> y) noexcept
	{
		T tmp_x[Vector<T>::length];
		T tmp_y[Vector<T>::length];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp_x[i] = std::pow(tmp_x[i], tmp_y[i]);
		return Vector<T>(tmp_x);
	}
	template<typename T>
	static inline Vector<T> mod(Vector<T> x, Vector<T> y) noexcept
	{
		T tmp_x[Vector<T>::length];
		T tmp_y[Vector<T>::length];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp_x[i] = std::fmod(tmp_x[i], tmp_y[i]);
		return Vector<T>(tmp_x);
	}
	template<typename T>
	static inline Vector<T> exp(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::exp(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> log(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::log(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> tanh(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::tanh(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> expm1(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::expm1(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> log1p(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::log1p(tmp[i]);
		return Vector<T>(tmp);
	}

	template<typename T>
	static inline Vector<T> sin(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::sin(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> cos(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::cos(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> tan(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::length];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::length; i++)
			tmp[i] = std::tan(tmp[i]);
		return Vector<T>(tmp);
	}

	static inline Vector<float16> pow(Vector<float16> x, Vector<float16> y) noexcept
	{
		return pow(static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
	}
	static inline Vector<float16> mod(Vector<float16> x, Vector<float16> y) noexcept
	{
		return mod(static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
	}
	static inline Vector<float16> exp(Vector<float16> x) noexcept
	{
		return exp(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> log(Vector<float16> x) noexcept
	{
		return log(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> tanh(Vector<float16> x) noexcept
	{
		return tanh(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> expm1(Vector<float16> x) noexcept
	{
		return expm1(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> log1p(Vector<float16> x) noexcept
	{
		return log1p(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> sin(Vector<float16> x) noexcept
	{
		return sin(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> cos(Vector<float16> x) noexcept
	{
		return cos(static_cast<Vector<float>>(x));
	}
	static inline Vector<float16> tan(Vector<float16> x) noexcept
	{
		return tan(static_cast<Vector<float>>(x));
	}

	static inline Vector<bfloat16> pow(Vector<bfloat16> x, Vector<bfloat16> y) noexcept
	{
		return pow(static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
	}
	static inline Vector<bfloat16> mod(Vector<bfloat16> x, Vector<bfloat16> y) noexcept
	{
		return mod(static_cast<Vector<float>>(x), static_cast<Vector<float>>(y));
	}
	static inline Vector<bfloat16> exp(Vector<bfloat16> x) noexcept
	{
		return exp(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> log(Vector<bfloat16> x) noexcept
	{
		return log(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> tanh(Vector<bfloat16> x) noexcept
	{
		return tanh(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> expm1(Vector<bfloat16> x) noexcept
	{
		return expm1(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> log1p(Vector<bfloat16> x) noexcept
	{
		return log1p(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> sin(Vector<bfloat16> x) noexcept
	{
		return sin(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> cos(Vector<bfloat16> x) noexcept
	{
		return cos(static_cast<Vector<float>>(x));
	}
	static inline Vector<bfloat16> tan(Vector<bfloat16> x) noexcept
	{
		return tan(static_cast<Vector<float>>(x));
	}

} /* namespace Vector_NAMESPACE */

#endif /* VECTORS_VECTOR_FUNCTIONS_HPP_ */
