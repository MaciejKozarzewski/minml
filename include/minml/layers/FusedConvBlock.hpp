/*
 * FusedConvBlock.hpp
 *
 *  Created on: Mar 4, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_FUSEDCONVBLOCK_HPP_
#define MINML_LAYERS_FUSEDCONVBLOCK_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class DepthwiseConv2D;
	class Conv2D;
}

namespace ml
{

	class FusedConvBlock: public Layer
	{
		private:
			Tensor m_depthwise_conv_weights;
			Tensor m_depthwise_conv_bias;
			Tensor m_conv_1_weights;
			Tensor m_conv_1_bias;
			Tensor m_conv_2_weights;
			Tensor m_conv_2_bias;
		public:
			FusedConvBlock();
			FusedConvBlock(const DepthwiseConv2D &dwc, const Conv2D &conv1, const Conv2D &conv2);

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
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_FUSEDCONVBLOCK_HPP_ */
