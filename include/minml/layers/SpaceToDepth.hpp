/*
 * SpaceToDepth.hpp
 *
 *  Created on: Aug 15, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_SPACETODEPTH_HPP_
#define MINML_LAYERS_SPACETODEPTH_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class SpaceToDepth: public Layer
	{
			int m_patch_size_h, m_patch_size_w;
		public:
			SpaceToDepth(int patch_size_h, int patch_size_w);
			SpaceToDepth(int patch_size);
			Shape getOutputShape() const;
			std::string name() const;
			Json getConfig() const;
			std::unique_ptr<Layer> clone(const Json &config) const;
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_SPACETODEPTH_HPP_ */
