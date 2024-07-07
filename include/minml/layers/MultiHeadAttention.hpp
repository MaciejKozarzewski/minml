/*
 * MultiHeadAttention.hpp
 *
 *  Created on: Jun 12, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_MULTIHEADATTENTION_HPP_
#define MINML_LAYERS_MULTIHEADATTENTION_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class MultiHeadAttention: public Layer
	{
			int m_number_of_heads = 0;
			int m_positional_encoding_range = 0;
		public:
			MultiHeadAttention(int numberOfHeads, int positional_encoding_range);

			Shape getWeightShape() const;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			int getWorkspaceSize() const noexcept;
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_MULTIHEADATTENTION_HPP_ */
