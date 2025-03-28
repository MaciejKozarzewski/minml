/*
 * SqueezeAndExcitation.hpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_SQUEEZEANDEXCITATION_HPP_
#define MINML_LAYERS_SQUEEZEANDEXCITATION_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class Dense;
}

namespace ml
{

	class SqueezeAndExcitation: public Layer
	{
		private:
			Tensor m_dense_1_weights;
			Tensor m_dense_1_bias;
			Tensor m_dense_2_weights;
			Tensor m_dense_2_bias;
		public:
			SqueezeAndExcitation();
			SqueezeAndExcitation(const Dense &dense1, const Dense &dense2);

			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;
			Json saveParameters(SerializedObject &binary_data) const;
			void loadParameters(const Json &json, const SerializedObject &binary_data);

			std::unique_ptr<Layer> clone(const Json &config) const;

			int getWorkspaceSize() const noexcept;
			void changeContext(std::shared_ptr<Context> &context);
			void convertTo(DataType newType);

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_SQUEEZEANDEXCITATION_HPP_ */
