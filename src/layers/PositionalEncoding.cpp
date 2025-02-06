/*
 * PositionalEncoding.cpp
 *
 *  Created on: Oct 31, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/PositionalEncoding.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace
{
	ml::Tensor convert_to_2D(const ml::Tensor &t)
	{
		const int first_dim = t.firstDim();
		const int last_dim = t.shape().volumeWithoutFirstDim();
		return t.view(ml::Shape( { first_dim, last_dim }));
	}
	ml::Tensor flatten(const ml::Tensor &t)
	{
		return t.view(ml::Shape( { t.volume() }));
	}
}

namespace ml
{
	PositionalEncoding::PositionalEncoding() :
			Layer("linear")
	{
	}

	void PositionalEncoding::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape PositionalEncoding::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape PositionalEncoding::getBiasShape() const
	{
//		Shape tmp( { getInputShape().dim(1), getInputShape().dim(2), getInputShape().dim(3), getInputShape().dim(3) });
		Shape tmp = getInputShape();
		tmp.removeDim(0);
		return tmp;
	}

	std::string PositionalEncoding::name() const
	{
		return "PositionalEncoding";
	}
	Json PositionalEncoding::getConfig() const
	{
		return Layer::getConfig();
	}

	std::unique_ptr<Layer> PositionalEncoding::clone(const Json &config) const
	{
		std::unique_ptr<PositionalEncoding> result = std::make_unique<PositionalEncoding>();
		result->loadConfig(config);
		return result;
	}

	void PositionalEncoding::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		const Tensor flattened_input = convert_to_2D(input[0]);
		const Tensor flattened_bias = flatten(getBias().getParam());
		Tensor flattened_output = convert_to_2D(output);
		addBiasAct(context(), flattened_output, flattened_input, flattened_bias, ActivationType::LINEAR);
	}
	void PositionalEncoding::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		Tensor flattened_gradient_next = convert_to_2D(gradient_next);
		Tensor flattened_gradient_bias = flatten(getBias().getGradient());
		sumOverFirstDim(context(), flattened_gradient_bias, flattened_gradient_next, 0);
		gradient_prev[0].copyFrom(context(), gradient_next);
	}

} /* namespace ml */

