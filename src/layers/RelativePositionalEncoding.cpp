/*
 * RelativePositionalEncoding.cpp
 *
 *  Created on: Nov 12, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/RelativePositionalEncoding.hpp>
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
	RelativePositionalEncoding::RelativePositionalEncoding(int max_range) :
			Layer("linear")
	{
	}

	void RelativePositionalEncoding::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape RelativePositionalEncoding::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape RelativePositionalEncoding::getWeightShape() const
	{
		const int tmp = 2 * m_max_range - 1;
		return Shape( { tmp, tmp, getInputShape().lastDim() });
	}

	std::string RelativePositionalEncoding::name() const
	{
		return "RelativePositionalEncoding";
	}
	Json RelativePositionalEncoding::getConfig() const
	{
		Json result = Layer::getConfig();
		result["max_range"] = m_max_range;
		return result;
	}

	std::unique_ptr<Layer> RelativePositionalEncoding::clone(const Json &config) const
	{
		std::unique_ptr<RelativePositionalEncoding> result = std::make_unique<RelativePositionalEncoding>(config["max_range"].getInt());
		result->loadConfig(config);
		return result;
	}

	void RelativePositionalEncoding::forward(const std::vector<Tensor> &input, Tensor &output)
	{
	}
	void RelativePositionalEncoding::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

	}

} /* namespace ml */
