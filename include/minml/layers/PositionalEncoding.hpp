/*
 * PositionalEncoding.hpp
 *
 *  Created on: Oct 31, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_POSITIONALENCODING_HPP_
#define MINML_LAYERS_POSITIONALENCODING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class PositionalEncoding: public Layer
	{
		public:
			PositionalEncoding();

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_POSITIONALENCODING_HPP_ */
