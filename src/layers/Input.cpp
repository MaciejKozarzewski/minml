/*
 * InputLayer.cpp
 *
 *  Created on: Feb 22, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Input.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	Input::Input(const Shape &input_shape) :
			Layer()
	{
		m_input_shapes.push_back(input_shape);
	}

	void Input::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Input layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Input::getOutputShape() const
	{
		return m_input_shapes[0];
	}

	std::string Input::name() const
	{
		return "Input";
	}
	Json Input::getConfig() const
	{
		Json result = Layer::getConfig();
		result["input_shape"] = getInputShape().serialize();
		return result;
	}

	std::unique_ptr<Layer> Input::clone(const Json &config) const
	{
		std::unique_ptr<Input> result = std::make_unique<Input>(Shape(config["input_shape"]));
		result->m_activation = activationFromString(config["nonlinearity"]);
		result->m_dtype = typeFromString(config["dtype"].getString());
		return std::unique_ptr<Layer>(static_cast<Layer*>(result.release()));
	}

	void Input::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		activationForwardInPlace(context(), output, m_activation);
	}
	void Input::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		activationBackwardInPlace(context(), gradient_next, output, m_activation);
	}

} /* namespace ml */

