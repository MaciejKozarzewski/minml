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

	template<typename ComputeT, typename T>
	void kernel_average_pooling(const void *input, void *output, int stride, int rows, int columns) noexcept
	{
		std::unique_ptr<ComputeT[]> acc = std::make_unique<ComputeT[]>(columns);
		for (int j = 0; j < columns; j++)
			acc[j] = static_cast<ComputeT>(0);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				acc[j] += convert<T, ComputeT>(reinterpret_cast<const T*>(input)[i * stride + j]);

		for (int j = 0; j < columns; j++)
			reinterpret_cast<T*>(output)[j] = convert<ComputeT, T>(acc[j] / rows);
	}

	template<typename ComputeT, typename T>
	void kernel_channel_scaling(const void *input, void *output, const void *scales, int stride, int rows, int columns) noexcept
	{
		std::unique_ptr<ComputeT[]> local_scales = std::make_unique<ComputeT[]>(columns);
		for (int j = 0; j < columns; j++)
			local_scales[j] = convert<T, ComputeT>(reinterpret_cast<const T*>(scales)[j]);

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				const ComputeT x = convert<T, ComputeT>(reinterpret_cast<const T*>(input)[i * stride + j]);
				reinterpret_cast<T*>(output)[i * stride + j] = convert<ComputeT, T>(x * local_scales[j]);
			}
	}
}

namespace ml
{
	void average_pooling_def_1xN(const TensorFragment &input, TensorFragment &output) noexcept
	{
		switch (input.dtype())
		{
			default:
				break;
			case DTYPE_FLOAT16:
			{
				assert(output.is_fp16());
				kernel_average_pooling<float, float16>(input.data(), output.data(), input.stride(), input.rows(), input.columns());
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(output.is_fp32());
				kernel_average_pooling<float, float>(input.data(), output.data(), input.stride(), input.rows(), input.columns());
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
				assert(scales.is_fp16());
				assert(output.is_fp16());
				kernel_channel_scaling<float, float16>(input.data(), output.data(), scales.data(), input.stride(), input.rows(), input.columns());
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(scales.is_fp32());
				assert(output.is_fp32());
				kernel_channel_scaling<float, float>(input.data(), output.data(), scales.data(), input.stride(), input.rows(), input.columns());
				break;
			}
		}
	}

} /* namespace ml */
