/*
 * quantization.cpp
 *
 *  Created on: Feb 6, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/quantization.hpp>
#include <minml/core/Tensor.hpp>

#include <memory>
#include <cmath>

namespace
{
	using namespace ml;

	Tensor transpose(const Tensor &x)
	{
		assert(x.rank() == 2);
		Tensor result( { x.lastDim(), x.firstDim() }, x.dtype(), x.device());
		for (int i = 0; i < x.firstDim(); i++)
			for (int j = 0; j < x.lastDim(); j++)
				result.at( { j, i }) = x.at( { i, j });
		return result;
	}
	float get_max_abs_value(const Tensor &x, int row) noexcept
	{
		assert(x.rank() == 2);
		float result = 0.0f;
		for (int j = 0; j < x.lastDim(); j++)
			result = std::max(result, std::fabs((float) x.at( { row, j })));
		return result;
	}
}

namespace ml
{
	int32_t quantize(float x, int bits) noexcept
	{
		const float tmp = 1 << (bits - 1);
		return std::max(-tmp, std::min(tmp - 1.0f, std::round(x)));
	}
	DataType get_quantized_dtype(int bits) noexcept
	{
		if (bits <= 8)
			return DataType::INT8;
		if (bits <= 16)
			return DataType::INT16;
		if (bits <= 32)
			return DataType::INT32;
		return DataType::UNKNOWN;
	}

	WeightQuantizer quantize_weights(const Tensor &weights, const Tensor &bias, const AffineTransform &input_transform, const std::string &mode,
			int bits)
	{
		assert(mode == "per_first_dim" || mode == "per_last_dim");
		WeightQuantizer result;
		Tensor tmp;

		if (mode == "per_last_dim")
			tmp = transpose(weights.view( { weights.shape().volumeWithoutLastDim(), weights.lastDim() }));
		else
			tmp = weights.view( { weights.firstDim(), weights.shape().volumeWithoutFirstDim() });

		result.weights = Tensor( { tmp.firstDim(), tmp.lastDim() }, get_quantized_dtype(bits), Device::cpu());
		result.channel_scales = Tensor( { tmp.firstDim() }, "float32", Device::cpu());
		result.bias = zeros_like(result.channel_scales);

		for (int i = 0; i < tmp.firstDim(); i++)
		{
			const float channel_scale = static_cast<float>((1 << (bits - 1)) - 1) / get_max_abs_value(tmp, i);
			result.channel_scales.at( { i }) = input_transform.scale() / channel_scale;

			int32_t sum_q = 0;
			for (int j = 0; j < tmp.lastDim(); j++)
			{
				const int32_t q = quantize((float) tmp.at( { i, j }) * channel_scale, bits);
				sum_q += static_cast<int32_t>(q);
				result.weights.at( { i, j }) = q;
			}

			const float b = bias.isEmpty() ? 0.0f : bias.at( { i });
			result.bias.at( { i }) = input_transform.shift() / channel_scale * sum_q + b;
		}

		if (mode == "per_last_dim")
			result.weights = transpose(result.weights);

		result.weights.reshape(weights.shape());
		return result;
	}

} /* namespace ml */

