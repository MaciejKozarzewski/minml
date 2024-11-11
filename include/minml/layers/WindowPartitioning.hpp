/*
 * WindowPartitioning.hpp
 *
 *  Created on: Nov 9, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_WINDOWPARTITIONING_HPP_
#define MINML_LAYERS_WINDOWPARTITIONING_HPP_

#include <minml/layers/Layer.hpp>

#include <vector>

namespace ml
{

	class WindowPartitioning: public Layer
	{
			int m_window_size = 0;
			int m_window_shift = 0;
		public:
			WindowPartitioning(int window_size, int window_shift);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_WINDOWPARTITIONING_HPP_ */
