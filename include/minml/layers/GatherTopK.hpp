/*
 * GatherTopK.hpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_GATHERTOPK_HPP_
#define MINML_LAYERS_GATHERTOPK_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class GatherTopK: public Layer
	{
			int m_top_k = 0;
			Tensor m_indices_cache;
		public:
			GatherTopK(int top_k);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;
			void changeContext(std::shared_ptr<Context> &context);

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_GATHERTOPK_HPP_ */
