/*
 * MLP.hpp
 *
 *  Created on: May 7, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_MLP_HPP_
#define MINML_LAYERS_MLP_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class MLP: public Layer
	{
			int m_neurons = 0;
			bool m_use_weights = true;
			bool m_use_bias = true;
		public:
			MLP(int neurons, std::string activation = "linear");

			MLP& useWeights(bool b) noexcept;
			MLP& useBias(bool b) noexcept;
			bool isUsingWeights() const noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			int getWorkspaceSize() const noexcept;
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */



#endif /* MINML_LAYERS_MLP_HPP_ */
