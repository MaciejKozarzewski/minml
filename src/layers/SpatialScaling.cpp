/*
 * SpatialScaling.cpp
 *
 *  Created on: Apr 19, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/SpatialScaling.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	SpatialScaling::SpatialScaling() :
			Layer("linear")
	{
	}

	void SpatialScaling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw IllegalArgument(METHOD_NAME, "SpatialScaling layer expects two input shapes");
		if (shapes.at(1).rank() != 2)
			throw IllegalArgument(METHOD_NAME, "SpatialScaling layer expects second argument to be 2D tensor");
		if (shapes.at(1).firstDim() != shapes.at(0).firstDim())
			throw IllegalArgument(METHOD_NAME, "SpatialScaling layer expects batch dimensions to be equal");
		if (shapes.at(1).lastDim() != shapes.at(0).dim(1) * shapes.at(0).dim(2))
			throw IllegalArgument(METHOD_NAME, "SpatialScaling layer expects last dimensions to be equal");
		m_input_shapes = shapes;
	}
	Shape SpatialScaling::getOutputShape() const
	{
		if (m_input_shapes.size() != 2)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape(0);
	}

	std::string SpatialScaling::name() const
	{
		return "SpatialScaling";
	}

	std::unique_ptr<Layer> SpatialScaling::clone(const Json &config) const
	{
		std::unique_ptr<SpatialScaling> result = std::make_unique<SpatialScaling>();
		result->loadConfig(config);
		return result;
	}

	void SpatialScaling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);
		spatialScalingForward(context(), 1.0f, input[0], input[1], 0.0f, output);
	}
	void SpatialScaling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		spatialScalingBackward(context(), 1.0f, gradient_next, input[0], input[1], beta[0], gradient_prev[0], beta[1], gradient_prev[1]);
	}

} /* namespace ml */
