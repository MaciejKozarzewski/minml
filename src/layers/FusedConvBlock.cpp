/*
 * FusedConvBlock.cpp
 *
 *  Created on: Mar 4, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/FusedConvBlock.hpp>
#include <minml/layers/Conv2D.hpp>
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
	FusedConvBlock::FusedConvBlock()
	{
	}
	FusedConvBlock::FusedConvBlock(const DepthwiseConv2D &dwc, const Conv2D &conv1, const Conv2D &conv2) :
			m_depthwise_conv_weights(dwc.getWeights().getParam()),
			m_depthwise_conv_bias(dwc.getBias().getParam()),
			m_conv_1_weights(conv1.getWeights().getParam()),
			m_conv_1_bias(conv1.getBias().getParam()),
			m_conv_2_weights(conv2.getWeights().getParam()),
			m_conv_2_bias(conv2.getBias().getParam())
	{
	}

	Shape FusedConvBlock::getOutputShape() const
	{
		return getInputShape();
	}

	std::string FusedConvBlock::name() const
	{
		return "FusedConvBlock";
	}
	Json FusedConvBlock::getConfig() const
	{
		return Layer::getConfig();
	}
	Json FusedConvBlock::saveParameters(SerializedObject &binary_data) const
	{
		Json result = Layer::saveParameters(binary_data);
		result["depthwise_conv_weights"] = m_depthwise_conv_weights.serialize(binary_data);
		result["depthwise_conv_bias"] = m_depthwise_conv_bias.serialize(binary_data);
		result["conv_1_weights"] = m_conv_1_weights.serialize(binary_data);
		result["conv_1_bias"] = m_conv_1_bias.serialize(binary_data);
		result["conv_2_weights"] = m_conv_2_weights.serialize(binary_data);
		result["conv_2_bias"] = m_conv_2_bias.serialize(binary_data);
		return result;
	}
	void FusedConvBlock::loadParameters(const Json &json, const SerializedObject &binary_data)
	{
		m_depthwise_conv_weights.unserialize(json["depthwise_conv_weights"], binary_data);
		m_depthwise_conv_bias.unserialize(json["depthwise_conv_bias"], binary_data);
		m_conv_1_weights.unserialize(json["conv_1_weights"], binary_data);
		m_conv_1_bias.unserialize(json["conv_1_bias"], binary_data);
		m_conv_2_weights.unserialize(json["conv_2_weights"], binary_data);
		m_conv_2_bias.unserialize(json["conv_2_bias"], binary_data);
	}

	std::unique_ptr<Layer> FusedConvBlock::clone(const Json &config) const
	{
		std::unique_ptr<FusedConvBlock> result = std::make_unique<FusedConvBlock>();
		result->loadConfig(config);
		return result;
	}

	int FusedConvBlock::getWorkspaceSize() const noexcept
	{
		return getOutputShape().volume();
	}
	void FusedConvBlock::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_depthwise_conv_weights.moveTo(device());
		m_depthwise_conv_bias.moveTo(device());
		m_conv_1_weights.moveTo(device());
		m_conv_1_bias.moveTo(device());
		m_conv_2_weights.moveTo(device());
		m_conv_2_bias.moveTo(device());
	}
	void FusedConvBlock::convertTo(DataType newType)
	{
		Layer::convertTo(newType);
		m_depthwise_conv_weights.convertTo(context(), newType);
		m_depthwise_conv_bias.convertTo(context(), newType);
		m_conv_1_weights.convertTo(context(), newType);
		m_conv_1_bias.convertTo(context(), newType);
		m_conv_2_weights.convertTo(context(), newType);
		m_conv_2_bias.convertTo(context(), newType);
	}

	void FusedConvBlock::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		if (device().isCPU())
		{
			const int batch = input[0].dim(0);
			const int height = input[0].dim(1);
			const int width = input[0].dim(2);
			const int filters = input[0].dim(3);

			const int kernel_height = m_depthwise_conv_weights.dim(0);
			const int kernel_width = m_depthwise_conv_weights.dim(1);

			const int pad_h = -(kernel_height - 1) / 2;
			const int pad_w = -(kernel_width - 1) / 2;

			output.zeroall();
			for (int b = 0; b < batch; b++)
				for (int f = 0; f < filters; f++)
					for (int h = 0; h < height; h++)
						for (int w = 0; w < width; w++)
						{
							float acc = m_depthwise_conv_bias.isEmpty() ? 0.0f : m_depthwise_conv_bias.get( { f });
							for (int i = 0; i < kernel_height; i++)
								for (int j = 0; j < kernel_width; j++)
								{
									const int x = pad_h + h + i;
									const int y = pad_w + w + j;
									if (0 <= x and x < height and 0 <= y and y < width)
										acc += m_depthwise_conv_weights.get( { i, j, f }) * input[0].get( { b, x, y, f });
								}
							output.at( { b, h, w, f }) = acc;
						}

			Tensor tmp = m_workspace.lock()->view(output.shape());
			for (int b = 0; b < batch; b++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
						for (int out = 0; out < filters; out++)
						{
//							float acc = m_conv_1_bias.isEmpty() ? 0.0f : m_conv_1_bias.get( { out });
//							for (int in = 0; in < filters; in++)
//								acc += m_conv_1_weights.get( { out, 0, 0, in }) * output.get( { b, h, w, in });
//							tmp.at( { b, h, w, out }) = std::max(0.0f, acc);

							float max_w = 0.0f, max_o = 0.0f;
							for (int in = 0; in < filters; in++)
							{
								max_w = std::max(max_w, std::abs(m_conv_1_weights.get( { out, 0, 0, in })));
								max_o = std::max(max_o, std::abs(output.get( { b, h, w, in })));
							}

							const float scale_w = 2047.0f;
							const float scale_o = 2047.0f;
							int32_t acc = 0;
							for (int in = 0; in < filters; in++)
							{
								const int32_t qw = scale_w / max_w * m_conv_1_weights.get( { out, 0, 0, in });
								const int32_t qo = scale_o / max_o * output.get( { b, h, w, in });
								acc += qw * qo;
							}
							const float bias = m_conv_1_bias.isEmpty() ? 0.0f : m_conv_1_bias.get( { out });
							tmp.at( { b, h, w, out }) = std::max(0.0f, bias + acc * max_w * max_o / (scale_w * scale_o));
						}

			for (int b = 0; b < batch; b++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
						for (int out = 0; out < filters; out++)
						{
//							float acc = m_conv_2_bias.isEmpty() ? 0.0f : m_conv_2_bias.get( { out });
//							for (int in = 0; in < filters; in++)
//								acc += m_conv_2_weights.get( { out, 0, 0, in }) * tmp.get( { b, h, w, in });
//							output.at( { b, h, w, out }) = acc + input[0].get( { b, h, w, out });

							float max_w = 0.0f, max_o = 0.0f;
							for (int in = 0; in < filters; in++)
							{
								max_w = std::max(max_w, std::abs(m_conv_2_weights.get( { out, 0, 0, in })));
								max_o = std::max(max_o, std::abs(tmp.get( { b, h, w, in })));
							}

							const float scale_w = 2047.0f;
							const float scale_o = 2047.0f;
							int32_t acc = 0;
							for (int in = 0; in < filters; in++)
							{
								const int32_t qw = scale_w / max_w * m_conv_2_weights.get( { out, 0, 0, in });
								const int32_t qo = scale_o / max_o * tmp.get( { b, h, w, in });
								acc += qw * qo;
							}
							const float bias = m_conv_2_bias.isEmpty() ? 0.0f : m_conv_2_bias.get( { out });
							output.at( { b, h, w, out }) = bias + acc * max_w * max_o / (scale_w * scale_o) + input[0].get( { b, h, w, out });
						}
		}
		if (device().isCUDA())
		{
			const int batch = input[0].dim(0);
			const int height = input[0].dim(1);
			const int width = input[0].dim(2);
			const int channels = input[0].dim(3);

			depthwiseConvForward(context(), 1.0f, input[0], m_depthwise_conv_weights, 0.0f, output, m_depthwise_conv_bias);

			Tensor tmp1_matrix = output.view( { batch * height * width, channels });
			Tensor tmp2_matrix = m_workspace.lock()->view( { batch * height * width, channels });
			Tensor weight_matrix = m_conv_1_weights.view( { channels, channels });

			gemm_ex(context(), tmp2_matrix, 1, 'n', tmp1_matrix, 't', weight_matrix, 0, tmp2_matrix, m_conv_1_bias, ActivationType::RELU);

			Tensor input_matrix = input[0].view( { batch * height * width, channels });
			Tensor output_matrix = output.view( { batch * height * width, channels });
			weight_matrix = m_conv_2_weights.view( { channels, channels });

			gemm_ex(context(), output_matrix, 1, 'n', tmp2_matrix, 't', weight_matrix, 1, input_matrix, m_conv_2_bias, ActivationType::LINEAR);
		}
	}
	void FusedConvBlock::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		throw LogicError(METHOD_NAME, "FusedConvBlock is not a trainable layer");
	}

} /* namespace ml */
