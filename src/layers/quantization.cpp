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

	WeightQuantizer quantize_weights(const Tensor &weights, const Tensor &bias, const AffineTransform &input_transform, const std::string &mode)
	{
		assert(mode == "per_first_dim" || mode == "per_last_dim");
		WeightQuantizer result;
		Tensor tmp;

		if (mode == "per_last_dim")
			tmp = transpose(weights.view( { weights.shape().volumeWithoutLastDim(), weights.lastDim() }));
		else
			tmp = weights.view( { weights.firstDim(), weights.shape().volumeWithoutFirstDim() });

		result.weights = Tensor( { tmp.firstDim(), tmp.lastDim() }, "int8", Device::cpu());
		result.channel_scales = Tensor( { tmp.firstDim() }, "float32", Device::cpu());
		result.bias = zeros_like(result.channel_scales);

		for (int i = 0; i < tmp.firstDim(); i++)
		{
			const float channel_scale = 127.0f / get_max_abs_value(tmp, i);
			result.channel_scales.at( { i }) = input_transform.scale() / channel_scale;

			int32_t sum_q = 0;
			for (int j = 0; j < tmp.lastDim(); j++)
			{
				const int8_t q = quantize_to<int8_t>((float) tmp.at( { i, j }) * channel_scale);
				sum_q += static_cast<int32_t>(q);
				result.weights.at( { i, j }) = q;
			}

			const float b = bias.isEmpty() ? 0.0f : bias.at( { i });
			result.bias.at( { i }) = input_transform.shift() / channel_scale * sum_q + b;
		}

//		if (mode == "per_first_dim")
//		{
//			const int first_dim = weights.firstDim();
//			const int last_dim = weights.shape().volumeWithoutFirstDim();
//			result.channel_scales = Tensor(Shape( { first_dim }), "float32", Device::cpu());
//			Tensor tmp = weights.view( { first_dim, last_dim });
//
//			result.weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
//			result.bias = Tensor( { first_dim }, "float32", Device::cpu());
//			for (int i = 0; i < first_dim; i++)
//			{
//				float max_abs_value = 0.0f;
//				for (int j = 0; j < last_dim; j++)
//					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));
//
//				const float channel_scale = 127.0f / max_abs_value;
//				result.channel_scales.at( { i }) = channel_scale;
//
//				int sum_q = 0;
//				for (int j = 0; j < last_dim; j++)
//				{
//					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
//					sum_q += static_cast<int>(q);
//					result.weights.at( { i, j }) = q;
//				}
//
//				result.bias.at( { i }) = input_quantizer.shift * sum_q * channel_scale;
//			}
//		}
//		if (mode == "per_last_dim")
//		{
//			const int first_dim = weights.shape().volumeWithoutLastDim();
//			const int last_dim = weights.lastDim();
//			result.channel_scales = Tensor(Shape( { last_dim }), "float32", Device::cpu());
//			Tensor tmp = weights.view( { first_dim, last_dim });
//
//			result.weights = Tensor( { first_dim, last_dim }, "int8", Device::cpu());
//			result.bias = Tensor( { last_dim }, "float32", Device::cpu());
//			for (int j = 0; j < last_dim; j++)
//			{
//				float max_abs_value = 0.0f;
//				for (int i = 0; i < first_dim; i++)
//					max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));
//
//				const float channel_scale = 127.0f / max_abs_value;
//				result.channel_scales.at( { j }) = channel_scale;
//
//				int sum_q = 0;
//				for (int i = 0; i < first_dim; i++)
//				{
//					const int8_t q = symmetric_quantize(tmp.get( { i, j }), channel_scale);
//					sum_q += static_cast<int>(q);
//					result.weights.at( { i, j }) = q;
//				}
//				result.bias.at( { j }) = input_quantizer.shift * sum_q * channel_scale;
//			}
//		}

		if (mode == "per_last_dim")
			result.weights = transpose(result.weights);

		result.weights.reshape(weights.shape());
		return result;
	}

} /* namespace ml */

