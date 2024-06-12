/*
 * backend_utils.hpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BACKEND_BACKEND_UTILS_HPP_
#define MINML_BACKEND_BACKEND_UTILS_HPP_

#include <minml/backend/backend_types.h>

#include <cstddef>
#include <initializer_list>
#include <cassert>

namespace ml
{
	template<typename T>
	T* getPointer(void *ptr) noexcept
	{
		return reinterpret_cast<T*>(ptr);
	}
	template<typename T>
	const T* getPointer(const void *ptr) noexcept
	{
		return reinterpret_cast<const T*>(ptr);
	}

	[[maybe_unused]] static mlShape_t make_shape(std::initializer_list<int> dims) noexcept
	{
		assert(dims.size() <= 4);
		mlShape_t result;
		result.rank = dims.size();
		for (int i = 0; i < result.rank; i++)
			result.dim[i] = dims.begin()[i];
		return result;
	}

	[[maybe_unused]] static size_t size_of(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return 0;
			case DTYPE_FLOAT16:
				return 2;
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
				return 4;
		}
	}

	[[maybe_unused]] static int volume(const mlShape_t &shape) noexcept
	{
		if (shape.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < shape.rank; i++)
				result *= shape.dim[i];
			return result;
		}
	}

	[[maybe_unused]] static int get_first_dim(const mlShape_t &shape) noexcept
	{
		return shape.dim[0];
	}
	[[maybe_unused]] static int get_last_dim(const mlShape_t &shape) noexcept
	{
		if (shape.rank == 0)
			return 0;
		else
			return shape.dim[shape.rank - 1];
	}

	[[maybe_unused]] static int volume_without_first_dim(const mlShape_t &shape) noexcept
	{
		if (shape.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 1; i < shape.rank; i++)
				result *= shape.dim[i];
			return result;
		}
	}
	[[maybe_unused]] static int volume_without_last_dim(const mlShape_t &shape) noexcept
	{
		if (shape.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < shape.rank - 1; i++)
				result *= shape.dim[i];
			return result;
		}
	}
	[[maybe_unused]] static bool is_transpose(char c) noexcept
	{
		assert(c == 'T' || c == 't' || c == 'N' || c == 'n');
		return c == 'T' || c == 't';
	}

} /* namespace ml */

#endif /* MINML_BACKEND_BACKEND_UTILS_HPP_ */
