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
		const int last_dim = getInputShape().volume(m_axis);
		if (last_dim == 0)
			throw LogicError(METHOD_NAME, "softmax cannot be calculated over zero volume dimensions");
		Shape result = getInputShape();

		for (size_t i = 0; i < m_axis.size(); i++)
		{
			result.removeDim(m_axis[i]);
			result.insertDim(m_axis[i], 1);
		}
		result.insertDim(result.rank(), last_dim);
		result.removeDim(0); // remove batch dimension
		result.squeeze();
		result.insertDim(0, getInputShape().firstDim()); // restore batch dimension
		return result;
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
		result->loadConfig(config);
		return result;
	}

	void Softmax::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const int first_dim = output.shape().volumeWithoutLastDim();
		const int last_dim = output.shape().lastDim();

		const Tensor in = input[0].view( { first_dim, last_dim });
		Tensor out = output.view( { first_dim, last_dim });
		softmaxForward(context(), out, in);
	}
	void Softmax::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);

		gradient_prev[0].copyFrom(context(), gradient_next.view(input[0].shape()));
	}
}

