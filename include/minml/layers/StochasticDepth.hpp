/*
 * StochasticDepth.hpp
 *
 *  Created on: Feb 24, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_STOCHASTICDEPTH_HPP_
#define MINML_LAYERS_STOCHASTICDEPTH_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class StochasticDepth: public Layer
	{
			float m_drop_p = 0.0f;
			bool m_drop_flag = false;
		public:
			StochasticDepth(float drop_p);
			Shape getOutputShape() const;
			std::string name() const;
			Json getConfig() const;
			std::unique_ptr<Layer> clone(const Json &config) const;
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_STOCHASTICDEPTH_HPP_ */
