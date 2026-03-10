/*
 * LearnableScaling.cpp
 *
 *  Created on: Feb 24, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/LearnableScaling.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	LearnableScaling::LearnableScaling(const std::string &act, float initialScale, bool trainableScale) :
			Layer(act),
			m_initial_scale(initialScale),
			m_trainable_scale(trainableScale)
	{
	}

	Shape LearnableScaling::getOutputShape() const
	{
		return getInputShape();
	}
	Shape LearnableScaling::getWeightShape() const
	{
		return Shape( { getInputShape().lastDim() });
	}

	std::string LearnableScaling::name() const
	{
		return "LearnableScaling";
	}
	Json LearnableScaling::getConfig() const
	{
		Json result = Layer::getConfig();
		result["initial_scale"] = m_initial_scale;
		result["trainable_scale"] = m_trainable_scale;
		return result;
	}

	std::unique_ptr<Layer> LearnableScaling::clone(const Json &config) const
	{
		const bool flag = config.hasKey("trainable_scale") ? config["trainable_scale"].getBool() : true;
		std::unique_ptr<LearnableScaling> result = std::make_unique<LearnableScaling>(config["nonlinearity"], config["initial_scale"].getDouble(),
				flag);
		result->loadConfig(config);
		return result;
	}

	void LearnableScaling::init()
	{
		getWeights().getParam().setall(context(), m_initial_scale);
	}

	void LearnableScaling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		channelLearnableScalingForward(context(), 1.0f, input[0], m_activation, getWeights().getParam(), 0.0f, output);
	}
	void LearnableScaling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		if (m_trainable_scale)
			channelLearnableScalingBackward(context(), 1.0f, gradient_next, input[0], m_activation, getWeights().getParam(), beta[0],
					gradient_prev[0], 0.0f, getWeights().getGradient());
		else
			channelLearnableScalingForward(context(), 1.0f, gradient_next, m_activation, getWeights().getParam(), beta[0], gradient_prev[0]);
	}

} /* namespace ml */
