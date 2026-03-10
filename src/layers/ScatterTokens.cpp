/*
 * ScatterTopK.cpp
 *
 *  Created on: Feb 10, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/layers/ScatterTokens.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	ScatterTokens::ScatterTokens(int height, int width) :
			Layer(),
			m_height(height),
			m_width(width)
	{
	}
	void ScatterTokens::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw LogicError(METHOD_NAME, "gather top K must have 2 inputs");
		const int batch_size = shapes[0].dim(0);
		const int top_k = shapes[0].dim(1);
		const int experts = shapes[0].dim(2);
		const int channels = shapes[0].dim(3);

		if (shapes[1].dim(0) != batch_size)
			throw LogicError(METHOD_NAME, "batch size mismatch");
		if (shapes[1].dim(1) != 2)
			throw LogicError(METHOD_NAME, "indices and values size mismatch");
		if (shapes[1].dim(2) != experts)
			throw LogicError(METHOD_NAME, "number of experts mismatch");

		m_input_shapes = shapes;
	}
	Shape ScatterTokens::getOutputShape() const
	{
		const int batch_size = getInputShape(0).dim(0);
		const int channels = getInputShape(0).dim(3);
		return Shape( { batch_size, m_height, m_width, channels });
	}
	std::string ScatterTokens::name() const
	{
		return "ScatterTopK";
	}
	Json ScatterTokens::getConfig() const
	{
		Json result = Layer::getConfig();
		result["height"] = m_height;
		result["width"] = m_width;
		return result;
	}
	std::unique_ptr<Layer> ScatterTokens::clone(const Json &config) const
	{
		std::unique_ptr<ScatterTokens> result = std::make_unique<ScatterTokens>(config["height"].getInt(), config["width"].getInt());
		result->loadConfig(config);
		return result;
	}
	void ScatterTokens::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);
		scatterTokensForward(context(), input[0], input[1], 0.0f, output);
	}
	void ScatterTokens::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		assert(gradient_prev.size() == 2);
		scatterTokensBackward(context(), gradient_next, input[0], input[1], beta[0], gradient_prev[0], beta[1], gradient_prev[1]);
	}

} /* namespace ml */
