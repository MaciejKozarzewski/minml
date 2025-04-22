/*
 * ChannelAveragePooling.cpp
 *
 *  Created on: Apr 19, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/ChannelAveragePooling.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	ChannelAveragePooling::ChannelAveragePooling() :
			Layer("linear")
	{
	}

	void ChannelAveragePooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		if (shapes.at(0).rank() != 4)
			throw IllegalArgument(METHOD_NAME, "expects 4D tensor");
		m_input_shapes = shapes;
	}
	Shape ChannelAveragePooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().firstDim();
		const int height = getInputShape().dim(1);
		const int width = getInputShape().dim(2);
		return Shape( { batch_size, height * width });
	}

	std::string ChannelAveragePooling::name() const
	{
		return "ChannelAveragePooling";
	}

	std::unique_ptr<Layer> ChannelAveragePooling::clone(const Json &config) const
	{
		std::unique_ptr<ChannelAveragePooling> result = std::make_unique<ChannelAveragePooling>();
		result->loadConfig(config);
		return result;
	}

	void ChannelAveragePooling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		channelAveragePoolingForward(context(), 1.0f, input[0], 0.0f, output);
	}
	void ChannelAveragePooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		channelAveragePoolingBackward(context(), 1.0f, gradient_next, beta[0], gradient_prev[0]);
	}

} /* namespace ml */

