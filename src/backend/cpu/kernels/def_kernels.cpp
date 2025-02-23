/*
 * def_kernels.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Author: Maciej Kozarzewski
 */

#include "TensorFragment.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <memory>
#include <cinttypes>
#include <cassert>

namespace
{
	using namespace ml;

	using namespace ml;
	using namespace ml::cpu;

	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	float16 convert(float x) noexcept
	{
		return cpu::convert_fp32_to_fp16(x);
	}
	template<>
	float convert(float16 x) noexcept
	{
		return cpu::convert_fp16_to_fp32(x);
	}

	template<typename ComputeT, typename InT, typename OutT>
	void kernel_average_pooling(const void *input, void *output, int stride, int rows, int columns) noexcept
	{
		std::unique_ptr<ComputeT[]> acc = std::make_unique<ComputeT[]>(columns);
		for (int j = 0; j < columns; j++)
			acc[j] = static_cast<ComputeT>(0);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				acc[j] += convert<InT, ComputeT>(reinterpret_cast<const InT*>(input)[i * stride + j]);

		for (int j = 0; j < columns; j++)
			reinterpret_cast<OutT*>(output)[j] = convert<ComputeT, OutT>(acc[j] / rows);
	}

	template<typename ComputeT, typename InT, typename OutT, typename ScaleT>
	void kernel_channel_scaling(const void *input, void *output, const void *scales, int stride, int rows, int columns) noexcept
	{
		std::unique_ptr<ComputeT[]> local_scales = std::make_unique<ComputeT[]>(columns);
		for (int j = 0; j < columns; j++)
			local_scales[j] = convert<ScaleT, ComputeT>(reinterpret_cast<const ScaleT*>(scales)[j]);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				const ComputeT x = convert<InT, ComputeT>(reinterpret_cast<const InT*>(input)[i * stride + j]);
				reinterpret_cast<OutT*>(output)[j] = convert<ComputeT, OutT>(x * local_scales[j]);
			}
	}
}

namespace ml
{
	void average_pooling_def_1xN(const void *input, mlDataType_t input_dtype, void *output, mlDataType_t output_dtype, int stride, int rows,
			int columns) noexcept
	{
		switch (input_dtype)
		{
			default:
				break;
			case DTYPE_FLOAT16:
			{
				assert(output_dtype == DTYPE_FLOAT16 || output_dtype == DTYPE_FLOAT32);
				if (output_dtype == DTYPE_FLOAT16)
					kernel_average_pooling<float, float16, float16>(input, output, stride, rows, columns);
				else
					kernel_average_pooling<float, float16, float>(input, output, stride, rows, columns);
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(output_dtype == DTYPE_FLOAT16 || output_dtype == DTYPE_FLOAT32);
				if (output_dtype == DTYPE_FLOAT16)
					kernel_average_pooling<float, float, float16>(input, output, stride, rows, columns);
				else
					kernel_average_pooling<float, float, float>(input, output, stride, rows, columns);
				break;
			}
			case DTYPE_FLOAT64:
			{
				assert(output_dtype == DTYPE_FLOAT64);
				kernel_average_pooling<double, double, double>(input, output, stride, rows, columns);
				break;
			}
		}
	}
	void channel_scaling_def_1xN(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		switch (input.dtype())
		{
			default:
				break;
			case DTYPE_FLOAT16:
			{
				assert(scales.is_fp16() || scales.is_fp32());
				assert(output.is_fp16() || output.is_fp32());
				if (output.is_fp16())
					kernel_channel_scaling<float, float16, float16, float16>(input.data(), output.data(), scales.data(), input.stride(), input.rows(),
							input.columns());
				else
					kernel_channel_scaling<float, float16, float, float16>(input.data(), output.data(), scales.data(), input.stride(), input.rows(),
							input.columns());
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(scales.is_fp16() || scales.is_fp32());
				assert(output.is_fp16() || output.is_fp32());
				if (output.is_fp16())
					kernel_channel_scaling<float, float, float16, float>(input.data(), output.data(), scales.data(), input.stride(), input.rows(),
							input.columns());
				else
					kernel_channel_scaling<float, float, float, float>(input.data(), output.data(), scales.data(), input.stride(), input.rows(),
							input.columns());
				break;
			}
			case DTYPE_FLOAT64:
			{
				assert(scales.is_fp64());
				assert(output.is_fp64());
				kernel_channel_scaling<double, double, double, double>(input.data(), output.data(), scales.data(), input.stride(), input.rows(),
						input.columns());
				break;
			}
		}
	}

} /* namespace ml */
