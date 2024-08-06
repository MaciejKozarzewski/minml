/*
 * Multiply.cpp
 *
 *  Created on: Jul 30, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Multiply.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/core/math.hpp>

namespace ml
{
	Multiply::Multiply() :
			Layer("linear")
	{
	}

	void Multiply::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() < 2)
			throw LogicError(METHOD_NAME, "Multiply layer expects at least two inputs");

		for (size_t i = 1; i < shapes.size(); i++)
			if (shapes[0] != shapes[i])
				throw ShapeMismatch(METHOD_NAME, shapes[0], shapes[i]);
		m_input_shapes = shapes;
	}
	Shape Multiply::getOutputShape() const
	{
		if (m_input_shapes.size() == 0)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}

	std::string Multiply::name() const
	{
		return "Multiply";
	}

	std::unique_ptr<Layer> Multiply::clone(const Json &config) const
	{
		std::unique_ptr<Multiply> result = std::make_unique<Multiply>();
		result->m_dtype = typeFromString(config["dtype"].getString());
		return std::unique_ptr<Layer>(static_cast<Layer*>(result.release()));
	}

	void Multiply::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == m_input_shapes.size());
		assert(input.size() == 2);

		multiplyTensors(context(), output, input[0], input[1]);
	}
	void Multiply::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == m_input_shapes.size());
		assert(gradient_prev.size() == m_input_shapes.size());

		multiplyTensors(context(), gradient_prev[0], gradient_next, input[1]);
		multiplyTensors(context(), gradient_prev[1], input[0], gradient_next);
	}

} /* namespace ml */

