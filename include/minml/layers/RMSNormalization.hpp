/*
 * RMSNormalization.hpp
 *
 *  Created on: Jul 26, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_RMSNORMALIZATION_HPP_
#define MINML_LAYERS_RMSNORMALIZATION_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class RMSNormalization: public Layer
	{
			bool m_use_gamma = true;
		public:
			RMSNormalization(bool useGamma = true);

			RMSNormalization& useGamma(bool b) noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void init();
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_RMSNORMALIZATION_HPP_ */
