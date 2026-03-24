/*
 * ChannelShuffle.hpp
 *
 *  Created on: Mar 23, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_CHANNELSHUFFLE_HPP_
#define MINML_LAYERS_CHANNELSHUFFLE_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class ChannelShuffle: public Layer
	{
			int m_groups = 1;
			bool m_invert = false;
		public:
			ChannelShuffle(int groups, bool invert = false);

			Shape getOutputShape() const;

			std::string name() const;

			Json getConfig() const;
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_CHANNELSHUFFLE_HPP_ */
