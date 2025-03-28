/*
 * GlobalPooling.hpp
 *
 *  Created on: Jul 10, 2024
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
			GlobalPooling();

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_GLOBALPOOLING_HPP_ */
