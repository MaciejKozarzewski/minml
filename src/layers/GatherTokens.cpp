/*
 * GatherTokens.cpp
 *
 *  Created on: Feb 8, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/layers/GatherTokens.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	GatherTokens::GatherTokens() :
			Layer()
	{
	}
	void GatherTokens::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw LogicError(METHOD_NAME, "gather top K must have 2 inputs");
		const int batch_size = shapes[0].dim(0);
		const int height = shapes[0].dim(1);
		const int width = shapes[0].dim(2);
		const int channels = shapes[0].dim(3);

		if (batch_size != shapes[1].dim(0))
			throw LogicError(METHOD_NAME, "batch size mismatch");
		if (2 != shapes[1].dim(1))
			throw LogicError(METHOD_NAME, "indices and values size mismatch");
		const int experts = shapes[1].dim(2);
		const int capacity = shapes[1].dim(3);

		m_input_shapes = shapes;
	}
	Shape GatherTokens::getOutputShape() const
	{
		const int batch_size = getInputShape(0).firstDim();
		const int channels = getInputShape(0).lastDim();
		const int experts = getInputShape(1).dim(2);
		const int capacity = getInputShape(1).dim(3);
		return Shape( { batch_size, capacity, experts, channels });
	}
	std::string GatherTokens::name() const
	{
		return "GatherTokens";
	}
	Json GatherTokens::getConfig() const
	{
		Json result = Layer::getConfig();
		return result;
	}
	std::unique_ptr<Layer> GatherTokens::clone(const Json &config) const
	{
		std::unique_ptr<GatherTokens> result = std::make_unique<GatherTokens>();
		result->loadConfig(config);
		return result;
	}
	void GatherTokens::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);
		gatherTokensForward(context(), input[0], input[1], 0.0f, output);
	}
	void GatherTokens::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		assert(gradient_prev.size() == 2);
		gatherTokensBackward(context(), gradient_next, input[1], beta[0], gradient_prev[0]);
	}

} /* namespace ml */

