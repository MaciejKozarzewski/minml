/*
 * Identity.cpp
 *
 *  Created on: Mar 27, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Identity.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	Identity::Identity() :
			Layer()
	{
	}

	Shape Identity::getOutputShape() const
	{
		return m_input_shapes[0];
	}

	std::string Identity::name() const
	{
		return "Identity";
	}

	std::unique_ptr<Layer> Identity::clone(const Json &config) const
	{
		std::unique_ptr<Identity> result = std::make_unique<Identity>();
		result->loadConfig(config);
		return result;
	}

	void Identity::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		output.copyFrom(context(), input[0]);
	}
	void Identity::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		if (beta[0] == 0.0f)
			gradient_prev[0].copyFrom(context(), gradient_next);
		else
			addTensors(context(), 1.0f, gradient_next, beta[0], gradient_prev[0], 0.0f, gradient_prev[0]);
	}

} /* namespace ml */

