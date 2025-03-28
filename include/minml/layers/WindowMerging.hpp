/*
 * WindowMerging.hpp
 *
 *  Created on: Nov 10, 2024
 *      Author: maciek
 */

#ifndef MINML_LAYERS_WINDOWMERGING_HPP_
#define MINML_LAYERS_WINDOWMERGING_HPP_

#include <minml/layers/Layer.hpp>

#include <vector>

namespace ml
{

	class WindowMerging: public Layer
	{
			Shape m_dst_shape;
			int m_window_shift = 0;
		public:
			WindowMerging(const Shape &shape, int shift);

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

#endif /* MINML_LAYERS_WINDOWMERGING_HPP_ */
