/*
 * Multiply.hpp
 *
 *  Created on: Jul 30, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_MULTIPLY_HPP_
#define MINML_LAYERS_MULTIPLY_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Multiply: public Layer
	{
		public:
			Multiply();

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */



#endif /* MINML_LAYERS_MULTIPLY_HPP_ */
