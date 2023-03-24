/*
 * Softmax.cpp
 *
 *  Created on: Jan 21, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Softmax.hpp>
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
	Softmax::Softmax(const std::vector<int> &axis) :
			m_axis(axis)
	{
		if (axis.size() == 0)
			throw LogicError(METHOD_NAME, "axis list must not be empty");
		m_activation = ActivationType::SOFTMAX;
	}

	void Softmax::setActivationType(ActivationType act) noexcept
	{
	}

	void Softmax::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Softmax layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Softmax::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return m_input_shapes[0];
	}

	std::string Softmax::name() const
	{
		return "Softmax";
	}
	Json Softmax::getConfig() const
	{
		Json result = Layer::getConfig();
		result["axis"] = Json(m_axis.data(), m_axis.size());
		return result;
	}

	std::unique_ptr<Layer> Softmax::clone(const Json &config) const
	{
		std::vector<int> axis;
		for (int i = 0; i < config["axis"].size(); i++)
			axis.push_back(config["axis"][i]);
		auto result = std::make_unique<Softmax>(axis);
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void Softmax::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const int last_dim = input[0].shape().volume(m_axis);
		if (last_dim == 0)
			throw LogicError(METHOD_NAME, "softmax cannot be calculated over zero volume dimensions");
		const int first_dim = input[0].shape().volume() / last_dim;
		const Shape shape( { first_dim, last_dim });

		const Tensor in = input[0].view(shape);
		Tensor out = output.view(shape);

		activationForward(context(), out, in, ActivationType::SOFTMAX);
	}
	void Softmax::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);

		gradient_prev[0].copyFrom(context(), gradient_next);
	}
}

