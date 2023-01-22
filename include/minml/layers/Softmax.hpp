/*
 * Softmax.hpp
 *
 *  Created on: Jan 21, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_SOFTMAX_HPP_
#define MINML_LAYERS_SOFTMAX_HPP_

#include <minml/layers/Layer.hpp>

#include <vector>

namespace ml
{

	class Softmax: public Layer
	{
			std::vector<int> m_axis;
		public:
			Softmax(const std::vector<int> &axis);

			void setActivationType(ActivationType act) noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_SOFTMAX_HPP_ */
