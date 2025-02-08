/*
 * ChannelScaling.cpp
 *
 *  Created on: Feb 7, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/ChannelScaling.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	ChannelScaling::ChannelScaling() :
			Layer("linear")
	{
	}

	void ChannelScaling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw IllegalArgument(METHOD_NAME, "ChannelScaling layer expects two input shapes");
		if (shapes.at(1).rank() != 2)
			throw IllegalArgument(METHOD_NAME, "ChannelScaling layer expects second argument to be 2D tensor");
		if (shapes.at(0).firstDim() != shapes.at(1).firstDim())
			throw IllegalArgument(METHOD_NAME, "ChannelScaling layer expects batch dimensions to be equal");
		if (shapes.at(0).lastDim() != shapes.at(1).lastDim())
			throw IllegalArgument(METHOD_NAME, "ChannelScaling layer expects last dimensions to be equal");
		m_input_shapes = shapes;
	}
	Shape ChannelScaling::getOutputShape() const
	{
		if (m_input_shapes.size() != 2)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape(0);
	}

	std::string ChannelScaling::name() const
	{
		return "ChannelScaling";
	}

	std::unique_ptr<Layer> ChannelScaling::clone(const Json &config) const
	{
		std::unique_ptr<ChannelScaling> result = std::make_unique<ChannelScaling>();
		result->loadConfig(config);
		return result;
	}

	void ChannelScaling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);
		output.copyFrom(context(), input[0]);
		channelScalingForward(context(), input[0], output, input[1]);
	}
	void ChannelScaling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 2);
		gradient_prev[0].copyFrom(context(), gradient_next);
		gradient_prev[1].zeroall(context());
		channelScalingBackward(context(), gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
	}

} /* namespace ml */
