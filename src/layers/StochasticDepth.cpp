/*
 * StochasticDepth.cpp
 *
 *  Created on: Feb 24, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/StochasticDepth.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/random.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	StochasticDepth::StochasticDepth(float drop_p) :
			Layer(),
			m_drop_p(drop_p)
	{
	}
	Shape StochasticDepth::getOutputShape() const
	{
		return getInputShape();
	}
	std::string StochasticDepth::name() const
	{
		return "StochasticDepth";
	}
	Json StochasticDepth::getConfig() const
	{
		Json result = Layer::getConfig();
		result["drop_p"] = m_drop_p;
		return result;
	}
	std::unique_ptr<Layer> StochasticDepth::clone(const Json &config) const
	{
		std::unique_ptr<StochasticDepth> result = std::make_unique<StochasticDepth>(config["drop_p"].getDouble());
		result->loadConfig(config);
		return result;
	}
	void StochasticDepth::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		m_drop_flag = isTrainable() ? (randFloat() < m_drop_p) : false;
		if (m_drop_flag)
			output.zeroall(context());
		else
			output.copyFrom(context(), input[0]);
	}
	void StochasticDepth::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		if (m_drop_flag)
			gradient_prev[0].zeroall(context());
		else
			gradient_prev[0].copyFrom(context(), gradient_next);
	}

} /* namespace ml */

