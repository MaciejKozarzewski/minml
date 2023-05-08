/*
 * Add.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: maciek
 */

#ifndef MINML_LAYERS_ADD_HPP_
#define MINML_LAYERS_ADD_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Add: public Layer
	{
		public:
			Add(std::string activation = "linear");

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_ADD_HPP_ */
