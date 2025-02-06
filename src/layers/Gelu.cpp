/*
 * Gelu.cpp
 *
 *  Created on: Nov 3, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Gelu.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/testing_util.hpp>

#include <initializer_list>
#include <memory>
#include <string>

namespace ml
{
	Gelu::Gelu()
	{
	}

	void Gelu::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Softmax layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Gelu::getOutputShape() const
	{
		return getInputShape();
	}

	std::string Gelu::name() const
	{
		return "Gelu";
	}

	std::unique_ptr<Layer> Gelu::clone(const Json &config) const
	{
		auto result = std::make_unique<Gelu>();
		result->loadConfig(config);
		return result;
	}

	void Gelu::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		activationForward(context(), output, input[0], ActivationType::GELU);
	}
	void Gelu::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);

		geluBackward(context(), gradient_prev[0], gradient_next, input[0]);
	}
}

