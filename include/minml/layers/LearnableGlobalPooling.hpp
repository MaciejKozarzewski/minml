/*
 * LearnableGlobalPooling.hpp
 *
 *  Created on: Mar 7, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_LEARNABLEGLOBALPOOLING_HPP_
#define MINML_LAYERS_LEARNABLEGLOBALPOOLING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class LearnableGlobalPooling: public Layer
	{
			int m_expansion_ratio;
		public:
			LearnableGlobalPooling(int expansionRatio, std::string activation = "linear");

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;

			Json getConfig() const;
			std::unique_ptr<Layer> clone(const Json &config) const;

			int getWorkspaceSize() const noexcept;
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_LEARNABLEGLOBALPOOLING_HPP_ */
