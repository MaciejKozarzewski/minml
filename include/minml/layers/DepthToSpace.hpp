/*
 * DepthToSpace.hpp
 *
 *  Created on: Aug 15, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_DEPTHTOSPACE_HPP_
#define MINML_LAYERS_DEPTHTOSPACE_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class DepthToSpace: public Layer
	{
			int m_patch_size_h, m_patch_size_w;
			Shape m_output_shape;
		public:
			DepthToSpace(int patch_size_h, int patch_size_w, const Shape &output_shape);
			DepthToSpace(int patch_size, const Shape &output_shape);
			Shape getOutputShape() const;
			std::string name() const;
			Json getConfig() const;
			std::unique_ptr<Layer> clone(const Json &config) const;
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_DEPTHTOSPACE_HPP_ */
