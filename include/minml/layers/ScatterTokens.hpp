/*
 * ScatterTokens.hpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_SCATTERTOKENS_HPP_
#define MINML_LAYERS_SCATTERTOKENS_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class ScatterTokens: public Layer
	{
			int m_height = 0;
			int m_width = 0;
		public:
			ScatterTokens(int height, int width);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_SCATTERTOKENS_HPP_ */
