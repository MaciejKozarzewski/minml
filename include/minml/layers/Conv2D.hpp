/*
 * Conv2D.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_CONV2D_HPP_
#define MINML_LAYERS_CONV2D_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{
	enum class ConvolutionAlgorithm
	{
		EXPLICIT_GEMM,
		IMPLICIT_GEMM,
		WINOGRAD_NON_FUSED
	};

	class Conv2D: public Layer
	{
		private:
			int m_output_filters = 0;
			int m_height = 0;
			int m_width = 0;
			int m_input_filters = 0;
			int m_kernel_size = 0;
			mutable ConvolutionAlgorithm m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
			mutable ConvolutionAlgorithm m_backward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
			mutable int m_winograd_tile_size = 1;
			std::unique_ptr<Tensor> m_transformed_weights;
			bool m_use_bias = true;
			bool m_are_weights_transformed = false;

		public:
			Conv2D(int filters, int kernelSize, std::string activation = "linear", bool useBias = true);

			Conv2D& useBias(bool b) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;
			void init();

			int getWorkspaceSize() const noexcept;
			void changeContext(std::shared_ptr<Context> &context);
			void invalidateWeightsCache();
			void convertTo(DataType newType);

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);

		private:
			void choose_algorithm() const;
	};

} /* namespace ml */

#endif /* MINML_LAYERS_CONV2D_HPP_ */
