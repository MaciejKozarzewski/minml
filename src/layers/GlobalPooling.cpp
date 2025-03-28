/*
 * GlobalPooling.cpp
 *
 *  Created on: Jul 10, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/GlobalPooling.hpp>
#include <minml/utils/json.hpp>

#include <minml/utils/time_util.hpp>

namespace ml
{
	GlobalPooling::GlobalPooling() :
			Layer("linear")
	{
	}

	void GlobalPooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape GlobalPooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().firstDim();
		const int channels = getInputShape().lastDim();
		return Shape( { batch_size, 2 * channels });
	}

	std::string GlobalPooling::name() const
	{
		return "GlobalPooling";
	}

	std::unique_ptr<Layer> GlobalPooling::clone(const Json &config) const
	{
		std::unique_ptr<GlobalPooling> result = std::make_unique<GlobalPooling>();
		result->loadConfig(config);
		return result;
	}

	void GlobalPooling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		globalAvgAndMaxPoolingForward(context(), input[0], output);
	}
	void GlobalPooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		globalAvgAndMaxPoolingBackward(context(), gradient_prev[0], gradient_next, input[0], output, beta[0]);
	}

} /* namespace ml */

