/*
 * conversion.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"

namespace
{
	using namespace SIMD_NAMESPACE;

	template<typename SrcType, typename DstType>
	struct Converter
	{
			void operator()(DstType *dst, const SrcType *src, int elements) const noexcept
			{
				Vector<float16>(src, elements).store(dst, elements);
			}
	};

//	template<typename T>
//	struct Converter<T, T>
//	{
//			void operator()(T *dst, const T *src, int elements) const noexcept
//			{
//				Vector<T>(src, elements).store(dst, elements);
//			}
//	};
//
//	template<>
//	struct Converter<float, float16>
//	{
//			void operator()(float16 *dst, const float *src, int elements) const noexcept
//			{
//				Vector<float>(src, elements).store(dst, elements);
//			}
//	};
//	template<>
//	struct Converter<float, bfloat16>
//	{
//			void operator()(bfloat16 *dst, const float *src, int elements) const noexcept
//			{
//				Vector<float>(src, elements).store(dst, elements);
//			}
//	};
//
//	template<>
//	struct Converter<float16, float>
//	{
//			void operator()(float *dst, const float16 *src, int elements) const noexcept
//			{
//				Vector<float>(src, elements).store(dst, elements);
//			}
//	};
//	template<>
//	struct Converter<float16, bfloat16>
//	{
//			void operator()(bfloat16 *dst, const float *src, int elements) const noexcept
//			{
//				Vector<bfloat16>(src, elements).store(dst, elements);
//			}
//	};
}

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

