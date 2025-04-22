/*
 * AveragePooling.cpp
 *
 *  Created on: Apr 21, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/AveragePooling.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	AveragePooling::AveragePooling(int size) :
			Layer("linear")
	{
		m_size = size;
	}

	void AveragePooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape AveragePooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().dim(0);
		const int height = getInputShape().dim(1);
		const int width = getInputShape().dim(2);
		const int channels = getInputShape().dim(3);
		return Shape( { batch_size, (height + m_size - 1) / m_size, (width + m_size - 1) / m_size, channels });
	}

	std::string AveragePooling::name() const
	{
		return "AveragePooling";
	}
	Json AveragePooling::getConfig() const
	{
		Json result = Layer::getConfig();
		result["size"] = m_size;
		return result;
	}

	std::unique_ptr<Layer> AveragePooling::clone(const Json &config) const
	{
		std::unique_ptr<AveragePooling> result = std::make_unique<AveragePooling>(config["size"].getInt());
		result->loadConfig(config);
		return result;
	}

	void AveragePooling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		averagePoolingForward(context(), 1.0f, input[0], 0.0f, output, m_size);
	}
	void AveragePooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		averagePoolingBackward(context(), 1.0f, gradient_next, beta[0], gradient_prev[0], m_size);
	}

} /* namespace ml */

