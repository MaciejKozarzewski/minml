/*
 * Gelu.hpp
 *
 *  Created on: Nov 3, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_GELU_HPP_
#define MINML_LAYERS_GELU_HPP_

#include <minml/layers/Layer.hpp>

#include <vector>

namespace ml
{

	class Gelu: public Layer
	{
		public:
			Gelu();

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_GELU_HPP_ */
