/*
 * GlobalAveragePooling.cpp
 *
 *  Created on: Feb 7, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/GlobalAveragePooling.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	GlobalAveragePooling::GlobalAveragePooling() :
			Layer("linear")
	{
	}

	void GlobalAveragePooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape GlobalAveragePooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().firstDim();
		const int channels = getInputShape().lastDim();
		return Shape( { batch_size, channels });
	}

	std::string GlobalAveragePooling::name() const
	{
		return "GlobalAveragePooling";
	}

	std::unique_ptr<Layer> GlobalAveragePooling::clone(const Json &config) const
	{
		std::unique_ptr<GlobalAveragePooling> result = std::make_unique<GlobalAveragePooling>();
		result->loadConfig(config);
		return result;
	}

	void GlobalAveragePooling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		globalAveragePoolingForward(context(), 1.0f, input[0], 0.0f, output);
	}
	void GlobalAveragePooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		globalAveragePoolingBackward(context(), 1.0f, gradient_next, beta[0], gradient_prev[0]);
	}

} /* namespace ml */

