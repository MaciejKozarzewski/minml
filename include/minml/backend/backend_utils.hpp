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
		assert(dims.size() <= 6);
		mlShape_t result;
		result.rank = dims.size();
		for (int i = 0; i < result.rank; i++)
			result.dim[i] = dims.begin()[i];
		return result;
	}

	[[maybe_unused]] static mlQuantizationData_t make_quantization_data(float scale, float shift) noexcept
	{
		mlQuantizationData_t result;
		result.scale = scale;
		result.shift = shift;
		return result;
	}

	[[maybe_unused]] static size_t size_of(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return 0;
			case DTYPE_FLOAT8:
			case DTYPE_UINT8:
			case DTYPE_INT8:
				return 1;
			case DTYPE_FLOAT16:
			case DTYPE_INT16:
				return 2;
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
				return 4;
			case DTYPE_FLOAT64:
				return 8;
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

	/*
	 * mlTensor_t helpers
	 */
	template<typename T>
	T* data(mlTensor_t &tensor) noexcept
	{
		return reinterpret_cast<T*>(tensor.data);
	}
	template<typename T>
	const T* data(const mlTensor_t &tensor) noexcept
	{
		return reinterpret_cast<const T*>(tensor.data);
	}
	[[maybe_unused]] static bool is_empty(const mlTensor_t &tensor) noexcept
	{
		return tensor.data == nullptr;
	}
	[[maybe_unused]] static bool is_fp16(const mlTensor_t &tensor) noexcept
	{
		return tensor.dtype == DTYPE_FLOAT16;
	}
	[[maybe_unused]] static bool is_fp32(const mlTensor_t &tensor) noexcept
	{
		return tensor.dtype == DTYPE_FLOAT32;
	}
	[[maybe_unused]] static int volume(const mlTensor_t &tensor) noexcept
	{
		if (tensor.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < tensor.rank; i++)
				result *= tensor.dim[i];
			return result;
		}
	}
	[[maybe_unused]] static int get_first_dim(const mlTensor_t &tensor) noexcept
	{
		if (tensor.rank == 0)
			return 0;
		else
			return tensor.dim[0];
	}
	[[maybe_unused]] static int get_last_dim(const mlTensor_t &tensor) noexcept
	{
		if (tensor.rank == 0)
			return 0;
		else
			return tensor.dim[tensor.rank - 1];
	}

	[[maybe_unused]] static int volume_without_first_dim(const mlTensor_t &tensor) noexcept
	{
		if (tensor.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 1; i < tensor.rank; i++)
				result *= tensor.dim[i];
			return result;
		}
	}
	[[maybe_unused]] static int volume_without_last_dim(const mlTensor_t &tensor) noexcept
	{
		if (tensor.rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < tensor.rank - 1; i++)
				result *= tensor.dim[i];
			return result;
		}
	}
	[[maybe_unused]] static mlTensor_t empty_tensor() noexcept
	{
		mlTensor_t result;
		result.data = nullptr;
		result.dtype = DTYPE_UNKNOWN;
		result.rank = 0;
		for (int i = 0; i < 6; i++)
			result.dim[i] = 0;
		return result;
	}
	[[maybe_unused]] static mlTensor_t make_tensor(const void *data, mlDataType_t dtype, const mlShape_t &shape) noexcept
	{
		mlTensor_t result;
		result.data = const_cast<void*>(data);
		result.dtype = dtype;
		result.rank = shape.rank;
		for (int i = 0; i < 6; i++)
			result.dim[i] = shape.dim[i];
		return result;
	}
	[[maybe_unused]] static mlTensor_t make_tensor(const void *data, mlDataType_t dtype, std::initializer_list<int> dims) noexcept
	{
		return make_tensor(data, dtype, make_shape(dims));
	}

	[[maybe_unused]] static int offset_at(const mlTensor_t &t, std::initializer_list<int> idx) noexcept
	{
		assert(idx.size() == static_cast<size_t>(t.rank));
		int stride = 1;
		int result = 0;
		for (int i = t.rank - 1; i >= 0; i--)
		{
			result += idx.begin()[i] * stride;
			stride *= t.dim[i];
		}
		return result;
	}

} /* namespace ml */

#endif /* MINML_BACKEND_BACKEND_UTILS_HPP_ */
