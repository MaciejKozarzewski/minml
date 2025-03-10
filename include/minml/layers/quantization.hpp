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
#include <limits>
#include <cmath>

namespace ml
{

	class AffineTransform
	{
			float m_scale = 1.0f;
			float m_shift = 0.0f;
		public:
			AffineTransform() noexcept = default;
			AffineTransform(float scale, float shift) noexcept :
					m_scale(scale),
					m_shift(shift)
			{
			}
			float scale() const noexcept
			{
				return m_scale;
			}
			float shift() const noexcept
			{
				return m_shift;
			}
			template<typename T>
			T operator()(T x) const noexcept
			{
				return static_cast<T>(static_cast<float>(x) * scale() + shift());
			}
			AffineTransform get_inverse() const noexcept
			{
				return AffineTransform(1.0f / scale(), -shift() / scale());
			}
			/*
			 * Returns combined transform of outer(inner())
			 */
			static AffineTransform combine(const AffineTransform &outer, const AffineTransform &inner) noexcept
			{
				return AffineTransform(outer.scale() * inner.scale(), outer.scale() * inner.shift() + outer.shift());
			}
	};

	template<typename T>
	T get_zero(const AffineTransform &t) noexcept
	{
		return static_cast<T>(std::round(t.get_inverse()(0.0f)));
	}

	class WeightQuantizer
	{
		public:
			Tensor weights;
			Tensor bias;
			Tensor channel_scales;
	};

	int32_t quantize(float x, int bits) noexcept;
	DataType get_quantized_dtype(int bits) noexcept;

	WeightQuantizer quantize_weights(const Tensor &weights, const Tensor &bias, const AffineTransform &input_transform, const std::string &mode,
			int bits = 8);

} /* namespace ml */

#endif /* MINML_LAYERS_QUANTIZATION_HPP_ */
