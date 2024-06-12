/*
 * LayerNormalization.hpp
 *
 *  Created on: May 7, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_LAYERNORMALIZATION_HPP_
#define MINML_LAYERS_LAYERNORMALIZATION_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class LayerNormalization: public Layer
	{
			bool m_use_gamma = true;
			bool m_use_beta = true;
		public:
			LayerNormalization(bool useGamma = true, bool useBeta = true);

			LayerNormalization& useGamma(bool b) noexcept;
			LayerNormalization& useBeta(bool b) noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void init();
			void setRegularizer(const Regularizer &regularizer);
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_LAYERNORMALIZATION_HPP_ */
