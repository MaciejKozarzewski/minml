/*
 * simd_functions.hpp
 *
 *  Created on: Dec 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_FUNCTIONS_HPP_
#define VECTORS_VECTOR_FUNCTIONS_HPP_

#include "vectors.hpp"

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
		T tmp_x[Vector<T>::size()];
		T tmp_y[Vector<T>::size()];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp_x[i] = std::pow(tmp_x[i], tmp_y[i]);
		return Vector<T>(tmp_x);
	}
	template<typename T>
	static inline Vector<T> mod(Vector<T> x, Vector<T> y) noexcept
	{
		T tmp_x[Vector<T>::size()];
		T tmp_y[Vector<T>::size()];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp_x[i] = std::fmod(tmp_x[i], tmp_y[i]);
		return Vector<T>(tmp_x);
	}
	template<typename T>
	static inline Vector<T> exp(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::exp(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> log(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::log(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> tanh(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::tanh(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> expm1(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::expm1(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> log1p(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::log1p(tmp[i]);
		return Vector<T>(tmp);
	}

	template<typename T>
	static inline Vector<T> sin(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::sin(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> cos(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::cos(tmp[i]);
		return Vector<T>(tmp);
	}
	template<typename T>
	static inline Vector<T> tan(Vector<T> x) noexcept
	{
		T tmp[Vector<T>::size()];
		x.store(tmp);
		for (int i = 0; i < Vector<T>::size(); i++)
			tmp[i] = std::tan(tmp[i]);
		return Vector<T>(tmp);
	}

} /* namespace Vector_NAMESPACE */

#endif /* VECTORS_VECTOR_FUNCTIONS_HPP_ */
