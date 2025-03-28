/*
 * Dequantize.hpp
 *
 *  Created on: Feb 3, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_DEQUANTIZE_HPP_
#define MINML_LAYERS_DEQUANTIZE_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Dequantize: public Layer
	{
			float m_scale = 1.0f;
			float m_shift = 0.0f;
		public:
			Dequantize(Shape input_shape = Shape());

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			void convertTo(DataType newType);
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_DEQUANTIZE_HPP_ */
