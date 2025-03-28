/*
 * DepthwiseConv2D.hpp
 *
 *  Created on: Jan 26, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_DEPTHWISECONV2D_HPP_
#define MINML_LAYERS_DEPTHWISECONV2D_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	class DepthwiseConv2D: public Layer
	{
		private:
			int m_filters = 0;
			int m_kernel_size = 0;
			bool m_use_bias = true;
		public:
			DepthwiseConv2D(int filters, int kernelSize, bool useBias = true);

			DepthwiseConv2D& useBias(bool b) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;
			void init();

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_DEPTHWISECONV2D_HPP_ */
