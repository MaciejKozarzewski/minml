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

	std::pair<Tensor, Tensor> WeightQuantizer::quantize(const Tensor &weights, const Tensor &bias, InputQuantizer input_quantizer, int mode)
	{
		Tensor quantized_weights;
		Tensor new_bias;
		if (mode == 0)
		{ // per first dim
			const int first_dim = weights.firstDim();
			const int last_dim = weights.shape().volumeWithoutFirstDim();
			m_channel_scales = std::make_unique<Tensor>(Shape( { first_dim }), "float32", Device::cpu());
			Tensor tmp = weights.view( { first_dim, last_dim });

			quantized_weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
			new_bias = Tensor( { first_dim }, "float32", Device::cpu());
			for (int i = 0; i < first_dim; i++)
			{
				float max_abs_value = 0.0f;
				for (int j = 0; j < last_dim; j++)
					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));

				const float channel_scale = 127.0f / max_abs_value;
				m_channel_scales->at( { i }) = channel_scale;

				int sum_q = 0;
				for (int j = 0; j < last_dim; j++)
				{
					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
					sum_q += static_cast<int>(q);
					quantized_weights.at( { i, j }) = q;
				}

				new_bias.at( { i }) = input_quantizer.shift * sum_q * channel_scale;
			}
		}
		if (mode == 1)
		{ // per last dim
			const int first_dim = weights.shape().volumeWithoutLastDim();
			const int last_dim = weights.lastDim();
			m_channel_scales = std::make_unique<Tensor>(Shape( { last_dim }), "float32", Device::cpu());
			Tensor tmp = weights.view( { first_dim, last_dim });

			quantized_weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
			new_bias = Tensor( { last_dim }, "float32", Device::cpu());
			for (int j = 0; j < last_dim; j++)
			{
				float max_abs_value = 0.0f;
				for (int i = 0; i < first_dim; i++)
					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));

				const float channel_scale = 127.0f / max_abs_value;
				m_channel_scales->at( { j }) = channel_scale;

				int sum_q = 0;
				for (int i = 0; i < first_dim; i++)
				{
					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
					sum_q += static_cast<int>(q);
					quantized_weights.at( { i, j }) = q;
				}
				new_bias.at( { j }) = input_quantizer.shift * sum_q * channel_scale;
			}
		}
		quantized_weights.reshape(weights.shape());

		if (not bias.isEmpty())
		{
			assert(bias.rank() == 1);
			for (int i = 0; i < bias.dim(0); i++)
				new_bias.at( { i }) = new_bias.get( { i }) + bias.get( { i });
		}

		return std::pair<Tensor, Tensor>(quantized_weights, new_bias);
	}
} /* namespace ml */

