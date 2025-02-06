/*
 * quantization.hpp
 *
 *  Created on: Feb 5, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_QUANTIZATION_HPP_
#define MINML_LAYERS_QUANTIZATION_HPP_

#include <memory>
#include <vector>

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Shape;
	class Tensor;
}

namespace ml
{
	class InputQuantizer
	{
			float inv_scale = 1.0f;
		public:
			float scale = 1.0f;
			float shift = 0.0f;

			InputQuantizer(float scale, float shift) noexcept :
					inv_scale(1.0f / scale),
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
				return std::max(-128.0f, std::min(127.0f, (x - shift) * inv_scale));
			}
	};

	class WeightQuantizer
	{
			std::unique_ptr<Tensor> m_channel_scales;
		public:
			std::pair<Tensor, Tensor> quantize(const Tensor &weights, const Tensor &bias, InputQuantizer input_quantizer, int mode);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_QUANTIZATION_HPP_ */
