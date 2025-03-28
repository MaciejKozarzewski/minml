/*
 * RelativePositionalEncoding.hpp
 *
 *  Created on: Nov 12, 2024
 *      Author: maciek
 */

#ifndef MINML_LAYERS_RELATIVEPOSITIONALENCODING_HPP_
#define MINML_LAYERS_RELATIVEPOSITIONALENCODING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class RelativePositionalEncoding: public Layer
	{
			int m_max_range = 0;
		public:
			RelativePositionalEncoding(int max_range);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_RELATIVEPOSITIONALENCODING_HPP_ */
