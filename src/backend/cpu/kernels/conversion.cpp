/*
 * conversion.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
	}
	void cpu_kernel_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_BFLOAT16:
						break;
					case DTYPE_FLOAT16:
						for (int i = 0; i < elements; i++)
							getPointer<bfloat16>(dst)[i] = scalar::float_to_bfloat16(scalar::float16_to_float(getPointer<float16>(src)[i]));
						break;
					case DTYPE_FLOAT32:
						for (int i = 0; i < elements; i++)
							getPointer<bfloat16>(dst)[i] = scalar::float_to_bfloat16(getPointer<float>(src)[i]);
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
						for (int i = 0; i < elements; i++)
							getPointer<float16>(dst)[i] = scalar::float_to_float16(scalar::bfloat16_to_float(getPointer<bfloat16>(src)[i]));
						break;
					case DTYPE_FLOAT16:
						break;
					case DTYPE_FLOAT32:
						for (int i = 0; i < elements; i++)
							getPointer<float16>(dst)[i] = scalar::float_to_float16(getPointer<float>(src)[i]);
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
						for (int i = 0; i < elements; i++)
							getPointer<float>(dst)[i] = scalar::bfloat16_to_float(getPointer<bfloat16>(src)[i]);
						break;
					case DTYPE_FLOAT16:
						for (int i = 0; i < elements; i++)
							getPointer<float>(dst)[i] = scalar::float16_to_float(getPointer<float16>(src)[i]);
						break;
					case DTYPE_FLOAT32:
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
}

