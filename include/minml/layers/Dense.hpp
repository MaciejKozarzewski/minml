/*
 * Dense.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_DENSE_HPP_
#define MINML_LAYERS_DENSE_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Dense: public Layer
	{
			int m_neurons = 0;
			bool m_use_bias = true;
		public:
			Dense(int neurons, const std::string &activation = "linear", bool useBias = true);

			Dense& useBias(bool b) noexcept;
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



#endif /* MINML_LAYERS_DENSE_HPP_ */
