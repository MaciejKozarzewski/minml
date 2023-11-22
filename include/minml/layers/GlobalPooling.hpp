/*
 * GlobalPooling.hpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_GLOBALPOOLING_HPP_
#define MINML_LAYERS_GLOBALPOOLING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class GlobalPooling: public Layer
	{
		public:
			GlobalPooling(std::string activation = "linear");

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;

			std::string name() const;

			int getWorkspaceSize() const noexcept;
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_GLOBALPOOLING_HPP_ */
