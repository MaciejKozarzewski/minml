/*
 * conversion.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"
#include "../vectors/vector_conversion.hpp"

#include <type_traits>

namespace
{
	using namespace SIMD_NAMESPACE;

	template<typename T, typename U>
	void kernel_convert(T *dst, const U *src, const int elements)
	{
		VectorConverter<T, U> convert;
		for (int i = 0; i < elements; i += convert.length)
		{
			const int processed_elements = std::min(convert.length, elements - i);
			const Vector<T> tmp = convert(Vector<U>(src + i, processed_elements));
			tmp.store(dst + i, processed_elements);
		}
	}

	template<typename T>
	void kernel_unpack_input(T *dst, const uint32_t *src, int first_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += 32)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < 32; j++, mask >>= 1)
				dst[j] = (mask & 1u) ? Vector<T>::scalar_one() : Vector<T>::scalar_zero();
		}
	}

	template<typename T>
	void transpose(void *dst, const void *src, int dim0, int dim1, int dim2)
	{
		for (int i = 0; i < dim0; i++)
			for (int j = 0; j < dim1; j++)
				for (int k = 0; k < dim2; k++)
				{
					const int src_idx = (i * dim1 + j) * dim2 + k;
					const int dst_idx = (i * dim2 + k) * dim1 + j;
					reinterpret_cast<T*>(dst)[dst_idx] = reinterpret_cast<const T*>(src)[src_idx];
				}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		assert(get_last_dim(shape) == 32);
		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
				kernel_unpack_input(getPointer<bfloat16>(dst), getPointer<uint32_t>(src), first_dim);
				break;
			case DTYPE_FLOAT16:
				kernel_unpack_input(getPointer<float16>(dst), getPointer<uint32_t>(src), first_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim);
				break;
			default:
				break;
		}
	}
	void cpu_kernel_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		if (dst_dtype == src_dtype and dst != src)
		{ // same type, different locations, can just copy memory
			std::memcpy(dst, src, size_of(dst_dtype) * elements);
			return;
		}

		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_FLOAT16:
						kernel_convert(getPointer<bfloat16>(dst), getPointer<float16>(src), elements);
						break;
					case DTYPE_FLOAT32:
						kernel_convert(getPointer<bfloat16>(dst), getPointer<float>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			case DTYPE_FLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_convert(getPointer<float16>(dst), getPointer<bfloat16>(src), elements);
						break;
					case DTYPE_FLOAT32:
						kernel_convert(getPointer<float16>(dst), getPointer<float>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			case DTYPE_FLOAT32:
			{
				switch (src_dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_convert(getPointer<float>(dst), getPointer<bfloat16>(src), elements);
						break;
					case DTYPE_FLOAT16:
						kernel_convert(getPointer<float>(dst), getPointer<float16>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			default:
				break;
		}
	}
	void cpu_kernel_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		assert(input != output);

		switch (dtype)
		{
			case DTYPE_BFLOAT16:
			case DTYPE_FLOAT16:
				transpose<uint16_t>(getPointer<uint16_t>(output), getPointer<uint16_t>(input), shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
				transpose<uint32_t>(getPointer<uint32_t>(output), getPointer<uint32_t>(input), shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			default:
				break;
		}
	}
}

