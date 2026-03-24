/*
 * ChannelShuffle.cpp
 *
 *  Created on: Mar 23, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/ChannelShuffle.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	ChannelShuffle::ChannelShuffle(int groups, bool invert) :
			Layer("linear"),
			m_groups(groups),
			m_invert(invert)
	{
	}

	Shape ChannelShuffle::getOutputShape() const
	{
		return getInputShape(0);
	}

	std::string ChannelShuffle::name() const
	{
		return "ChannelShuffle";
	}

	Json ChannelShuffle::getConfig() const
	{
		Json result = Layer::getConfig();
		result["groups"] = m_groups;
		result["invert"] = m_invert;
		return result;
	}
	std::unique_ptr<Layer> ChannelShuffle::clone(const Json &config) const
	{
		const bool invert = config.hasKey("invert") ? config["invert"].getBool() : false;
		std::unique_ptr<ChannelShuffle> result = std::make_unique<ChannelShuffle>(config["groups"].getInt(), invert);
		result->loadConfig(config);
		return result;
	}

	void ChannelShuffle::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		channelShuffle(context(), input[0], 0.0f, output, m_groups, m_invert);
	}
	void ChannelShuffle::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		channelShuffle(context(), gradient_next, beta[0], gradient_prev[0], m_groups, not m_invert);
	}

} /* namespace ml */
