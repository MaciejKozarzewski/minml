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
	int8_t symmetric_quantize(float x, float scale) noexcept
	{
		return std::max(-127.0f, std::min(127.0f, x * scale));
	}
}

namespace ml
{

	WeightQuantizer quantize_weights(const Tensor &weights, const Tensor &bias, const TensorQuantizer &input_quantizer, const std::string &mode)
	{
		assert(mode == "per_first_dim" || mode == "per_last_dim");
		WeightQuantizer result;
		if (mode == "per_first_dim")
		{
			const int first_dim = weights.firstDim();
			const int last_dim = weights.shape().volumeWithoutFirstDim();
			result.channel_scales = Tensor(Shape( { first_dim }), "float32", Device::cpu());
			Tensor tmp = weights.view( { first_dim, last_dim });

			result.weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
			result.bias = Tensor( { first_dim }, "float32", Device::cpu());
			for (int i = 0; i < first_dim; i++)
			{
				float max_abs_value = 0.0f;
				for (int j = 0; j < last_dim; j++)
					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));

				const float channel_scale = 127.0f / max_abs_value;
				result.channel_scales.at( { i }) = channel_scale;

				int sum_q = 0;
				for (int j = 0; j < last_dim; j++)
				{
					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
					sum_q += static_cast<int>(q);
					result.weights.at( { i, j }) = q;
				}

				result.bias.at( { i }) = input_quantizer.shift * sum_q * channel_scale;
			}
		}
		if (mode == "per_last_dim")
		{
			const int first_dim = weights.shape().volumeWithoutLastDim();
			const int last_dim = weights.lastDim();
			result.channel_scales = Tensor(Shape( { last_dim }), "float32", Device::cpu());
			Tensor tmp = weights.view( { first_dim, last_dim });

			result.weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
			result.bias = Tensor( { last_dim }, "float32", Device::cpu());
			for (int j = 0; j < last_dim; j++)
			{
				float max_abs_value = 0.0f;
				for (int i = 0; i < first_dim; i++)
					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));

				const float channel_scale = 127.0f / max_abs_value;
				result.channel_scales.at( { j }) = channel_scale;

				int sum_q = 0;
				for (int i = 0; i < first_dim; i++)
				{
					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
					sum_q += static_cast<int>(q);
					result.weights.at( { i, j }) = q;
				}
				result.bias.at( { j }) = input_quantizer.shift * sum_q * channel_scale;
			}
		}
		result.weights.reshape(weights.shape());

		if (not bias.isEmpty())
		{
			assert(bias.rank() == 1);
			for (int i = 0; i < bias.dim(0); i++)
				result.bias.at( { i }) = result.bias.get( { i }) + bias.get( { i });
		}

		return result;
	}
} /* namespace ml */

