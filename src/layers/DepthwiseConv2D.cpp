/*
 * DepthwiseConv2D.cpp
 *
 *  Created on: Jan 27, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/string_util.hpp>
#include <minml/utils/time_util.hpp>

#include <cmath>

namespace ml
{
	DepthwiseConv2D::DepthwiseConv2D(int filters, int kernelSize, bool useBias) :
			Layer("linear")
	{
		m_filters = filters;
		m_kernel_size = kernelSize;
		m_use_bias = useBias;
	}

	DepthwiseConv2D& DepthwiseConv2D::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool DepthwiseConv2D::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void DepthwiseConv2D::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "DepthwiseConv2D layer expects one input shape");
		if (shapes[0].rank() != 4)
			throw IllegalArgument(METHOD_NAME, "DepthwiseConv2D layer expects 4D shapes");

		m_input_shapes = shapes;
	}
	Shape DepthwiseConv2D::getOutputShape() const
	{
		return getInputShape();
	}
	Shape DepthwiseConv2D::getWeightShape() const
	{
		return Shape( { m_kernel_size, m_kernel_size, m_filters });
	}
	Shape DepthwiseConv2D::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_filters });
		else
			return Shape();
	}

	std::string DepthwiseConv2D::name() const
	{
		return "DepthwiseConv2D";
	}
	Json DepthwiseConv2D::getConfig() const
	{
		Json result = Layer::getConfig();
		result["filters"] = m_filters;
		result["kernel_size"] = m_kernel_size;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> DepthwiseConv2D::clone(const Json &config) const
	{
		std::unique_ptr<DepthwiseConv2D> result = std::make_unique<DepthwiseConv2D>(config["filters"], config["kernel_size"], config["use_bias"]);
		result->loadConfig(config);
		return result;
	}
	void DepthwiseConv2D::init()
	{
		m_initializer.init_weights(context(), getWeights(), 0.1f * std::sqrt(1.0f / m_filters), 0.0f);
		m_initializer.init_bias(context(), getBias(), 0.1f, 0.0f);
	}

	void DepthwiseConv2D::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		if (isInteger(input[0].dtype()))
		{
			assert(output.dtype() == dtype());
			if (device().isCUDA())
			{
				const int32_t input_zero = get_zero<int32_t>(m_input_transforms[0]);
				quantized_depthwise_conv_forward(context(), input[0], getWeights().getParam(), m_channel_scales, getBias().getParam(), output,
						m_output_transform, input_zero);
			}
			if (device().isCPU())
			{
				const int batch = input[0].dim(0);
				const int height = input[0].dim(1);
				const int width = input[0].dim(2);
				const int filters = input[0].dim(3);

				const int pad_h = -(m_kernel_size - 1) / 2;
				const int pad_w = -(m_kernel_size - 1) / 2;

				const int32_t input_zero = get_zero<int32_t>(m_input_transforms[0]);
				const AffineTransform output_to_int = m_output_transform.get_inverse();

				for (int b = 0; b < batch; b++)
					for (int f = 0; f < filters; f++)
						for (int h = 0; h < height; h++)
							for (int w = 0; w < width; w++)
							{
								int32_t acc = 0;
								for (int i = 0; i < m_kernel_size; i++)
									for (int j = 0; j < m_kernel_size; j++)
									{
										const int x = pad_h + h + i;
										const int y = pad_w + w + j;
										if (0 <= x and x < height and 0 <= y and y < width)
											acc += (int) getWeights().getParam().at( { i, j, f }) * (int) input[0].at( { b, x, y, f });
										else
											acc += (int) getWeights().getParam().at( { i, j, f }) * input_zero;
									}
								float tmp = static_cast<float>(acc) * (float) m_channel_scales.at( { f }) + (float) getBias().getParam().at( { f });

								if (isInteger(output.dtype()))
									output.at( { b, h, w, f }) = quantize(output_to_int(tmp), m_quantization_bits);
								if (output.dtype() == DataType::FLOAT32)
									output.at( { b, h, w, f }) = tmp;
							}
			}
		}
		else
			depthwiseConvForward(context(), 1.0f, input[0], getWeights().getParam(), 0.0f, output, getBias().getParam());
	}
	void DepthwiseConv2D::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1 && gradient_prev.size() == 1);

		Tensor empty;
		fusedBiasActCopyBackward(context(), gradient_next, output, 0.0f, empty, 0.0f, getBias().getGradient(), m_activation);

		depthwiseConvBackward(context(), 1.0f, gradient_next, getWeights().getParam(), beta[0], gradient_prev[0]);
		depthwiseConvUpdate(context(), 1.0f, input[0], gradient_next, 0.0f, getWeights().getGradient());
	}

} /* namespace ml */

