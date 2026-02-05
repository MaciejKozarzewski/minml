/*
 * AbsolutePositionalEncoding.cpp
 *
 *  Created on: Feb 2, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/AbsolutePositionalEncoding.hpp>
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
	AbsolutePositionalEncoding::AbsolutePositionalEncoding() :
			Layer("linear")
	{
	}

	void AbsolutePositionalEncoding::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape AbsolutePositionalEncoding::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape AbsolutePositionalEncoding::getBiasShape() const
	{
		Shape tmp = getInputShape();
		tmp.removeDim(0);
		return tmp;
	}

	std::string AbsolutePositionalEncoding::name() const
	{
		return "AbsolutePositionalEncoding";
	}
	Json AbsolutePositionalEncoding::getConfig() const
	{
		return Layer::getConfig();
	}

	std::unique_ptr<Layer> AbsolutePositionalEncoding::clone(const Json &config) const
	{
		std::unique_ptr<AbsolutePositionalEncoding> result = std::make_unique<AbsolutePositionalEncoding>();
		result->loadConfig(config);
		return result;
	}

	void AbsolutePositionalEncoding::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		const Tensor flattened_input = convert_to_2D(input[0]);
		const Tensor flattened_bias = flatten(getBias().getParam());
		Tensor flattened_output = convert_to_2D(output);
		addBiasAct(context(), 1.0f, flattened_input, flattened_bias, 0.0f, flattened_output, ActivationType::LINEAR);
	}
	void AbsolutePositionalEncoding::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		Tensor flattened_gradient_next = convert_to_2D(gradient_next);
		Tensor flattened_gradient_bias = flatten(getBias().getGradient());
		sumOverFirstDim(context(), 1.0f, flattened_gradient_next, 0.0f, flattened_gradient_bias);
		gradient_prev[0].copyFrom(context(), gradient_next);
	}

} /* namespace ml */

