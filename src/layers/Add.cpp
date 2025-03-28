/*
 * Add.cpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Add.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/core/math.hpp>

namespace ml
{
	Add::Add(std::string activation) :
			Layer(activation)
	{
	}

	void Add::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() < 2)
			throw LogicError(METHOD_NAME, "Add layer expects at least two inputs");

		for (size_t i = 1; i < shapes.size(); i++)
			if (shapes[0] != shapes[i])
				throw ShapeMismatch(METHOD_NAME, shapes[0], shapes[i]);
		m_input_shapes = shapes;
	}
	Shape Add::getOutputShape() const
	{
		if (m_input_shapes.size() == 0)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}

	std::string Add::name() const
	{
		return "Add";
	}

	std::unique_ptr<Layer> Add::clone(const Json &config) const
	{
		std::unique_ptr<Add> result = std::make_unique<Add>(config["nonlinearity"]);
		result->loadConfig(config);
		return result;
	}

	void Add::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == m_input_shapes.size());
		assert(input.size() == 2);

		addTensors(context(), output, input[0], input[1]);
		activationForward(context(), 1.0f, output, 0.0f, output, m_activation);
	}
	void Add::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == m_input_shapes.size());
		assert(gradient_prev.size() == m_input_shapes.size());

		activationBackward(context(), 1.0f, gradient_next, output, 0.0f, gradient_next, m_activation);
		for (size_t i = 0; i < gradient_prev.size(); i++)
			if (beta[i] == 0.0f)
				gradient_prev[i].copyFrom(context(), gradient_next);
			else
				addTensors(context(), 0.0f, gradient_prev[i], beta[i], gradient_prev[i], 1.0f, gradient_next);
	}

} /* namespace ml */

