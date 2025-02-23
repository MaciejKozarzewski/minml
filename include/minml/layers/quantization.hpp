/*
 * quantization.hpp
 *
 *  Created on: Feb 5, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_QUANTIZATION_HPP_
#define MINML_LAYERS_QUANTIZATION_HPP_

#include <minml/core/Tensor.hpp>

#include <string>

namespace ml
{
	class TensorQuantizer
	{
		public:
			float scale = 1.0f;
			float shift = 0.0f;

			TensorQuantizer() noexcept = default;
			TensorQuantizer(float scale, float shift) noexcept :
					scale(scale),
					shift(shift)
			{
			}
			float to_fp32(int8_t x) const noexcept
			{
				return static_cast<float>(x) * scale + shift;
			}
			int8_t to_int8(float x) const noexcept
			{
				return std::max(-128.0f, std::min(127.0f, (x - shift) / scale));
			}
	};

	class WeightQuantizer
	{
		public:
			Tensor weights;
			Tensor bias;
			Tensor channel_scales;
	};

	WeightQuantizer quantize_weights(const Tensor &weights, const Tensor &bias, const TensorQuantizer &input_quantizer, const std::string &mode);

} /* namespace ml */

#endif /* MINML_LAYERS_QUANTIZATION_HPP_ */
