/*
 * Router.hpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_ROUTER_HPP_
#define MINML_LAYERS_ROUTER_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Router: public Layer
	{
			int m_experts = 0;
		public:
			Router(int experts);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			int getWorkspaceSize() const noexcept;
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_ROUTER_HPP_ */
