/*
 * Router.cpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Router.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace
{
	using namespace ml;
	Shape get_tmp_output_shape(int experts, const Shape &input_shape)
	{
		assert(input_shape.rank() == 4);
		return Shape( { input_shape[0], input_shape[1] * input_shape[2], experts });
	}
	std::string algo_to_string(RoutingAlgorithm ra)
	{
		switch (ra)
		{
			case RoutingAlgorithm::HASH:
				return "hash";
			case RoutingAlgorithm::TOKEN_CHOICE:
				return "token_choice";
			case RoutingAlgorithm::EXPERT_CHOICE:
				return "expert_choice";
			default:
				return "";
		}
	}
	RoutingAlgorithm algo_from_string(const std::string &str)
	{
		if (str == "hash")
			return RoutingAlgorithm::HASH;
		if (str == "token_choice")
			return RoutingAlgorithm::TOKEN_CHOICE;
		if (str == "expert_choice")
			return RoutingAlgorithm::EXPERT_CHOICE;
		throw LogicError(METHOD_NAME, "unknown routing algorithm '" + str + "'");
	}
	Json tensor_to_json(const Tensor &t)
	{
		if (t.isEmpty())
			return Json::null();
		assert(t.rank() == 1);
		Json result = Json::array();
		for (int i = 0; i < t.volume(); i++)
			result[i] = (float) t.at( { i });
		return result;
	}
	Tensor tensor_from_json(const Json &json)
	{
		if (json.isNull())
			return Tensor();
		Tensor result(Shape( { json.size() }), DataType::FLOAT32, Device::cpu());
		for (int i = 0; i < result.volume(); i++)
			result.at( { i }) = json[i].getDouble();
		return result;
	}
	int get_capacity(float capacity_factor, int tokens, int experts) noexcept
	{
		const int tmp = static_cast<int>(capacity_factor * tokens / experts);
		return (tmp * experts >= tokens) ? tmp : (tmp + 1);
	}
}

namespace ml
{
	Router::Router(RoutingAlgorithm algo, float capacityFactor, float loadBalancingAlpha) :
			Layer(),
			m_algorithm(algo),
			m_capacity_factor(capacityFactor),
			m_load_balancing_alpha(loadBalancingAlpha)
	{
	}

	void Router::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "Router can only have 1 input");
		m_input_shapes = shapes;
	}
	Shape Router::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().dim(0);
		const int tokens = getInputShape().dim(1) * getInputShape().dim(2);
		const int experts = getInputShape().dim(3);
		const int capacity = get_capacity(m_capacity_factor, tokens, experts);

		return Shape( { batch_size, 2, experts, capacity });
	}

	std::string Router::name() const
	{
		return "Router";
	}
	Json Router::getConfig() const
	{
		Json result = Layer::getConfig();
		result["algorithm"] = algo_to_string(m_algorithm);
		result["capacity_factor"] = m_capacity_factor;
		result["load_balancing_alpha"] = m_load_balancing_alpha;
		result["expert_biases"] = tensor_to_json(m_expert_biases);
		return result;
	}

	void Router::convertTo(DataType newType)
	{
		Layer::convertTo(newType);
		m_expert_biases.convertTo(context(), newType);
	}
	std::unique_ptr<Layer> Router::clone(const Json &config) const
	{
		std::unique_ptr<Router> result = std::make_unique<Router>(algo_from_string(config["algorithm"].getString()),
				config["capacity_factor"].getDouble(), config["load_balancing_alpha"].getDouble());
		result->m_expert_biases = tensor_from_json(config["expert_biases"]);
		result->loadConfig(config);
		return result;
	}
	int Router::getWorkspaceSize() const noexcept
	{
		if (m_algorithm == RoutingAlgorithm::TOKEN_CHOICE)
			return getInputShape().volume();
		else
			return 0;
	}
	void Router::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_expert_biases.moveTo(context->device());
	}

	void Router::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		switch (m_algorithm)
		{
			case RoutingAlgorithm::HASH:
				hashRouting(context(), input[0], output);
				break;
			case RoutingAlgorithm::TOKEN_CHOICE:
			{
				if (m_expert_biases.isEmpty())
					m_expert_biases = Tensor( { getInputShape().dim(3) }, dtype(), device());
				tokenChoiceRoutingForward(context(), input[0], m_expert_biases, output);
				break;
			}
			case RoutingAlgorithm::EXPERT_CHOICE:
				expertChoiceRoutingForward(context(), input[0], output);
				break;
			default:
				break;
		}

	}
	void Router::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		Tensor workspace = m_workspace.lock()->view(gradient_prev[0].shape());
		switch (m_algorithm)
		{
			case RoutingAlgorithm::HASH:
				break;
			case RoutingAlgorithm::TOKEN_CHOICE:
			{
				if (m_expert_biases.isEmpty())
					m_expert_biases = Tensor( { getInputShape().dim(3) }, dtype(), device());
				tokenChoiceRoutingBackward(context(), input[0], output, gradient_next, beta[0], gradient_prev[0], m_load_balancing_alpha,
						m_expert_biases, workspace);
				break;
			}
			case RoutingAlgorithm::EXPERT_CHOICE:
				expertChoiceRoutingBackward(context(), input[0], output, gradient_next, beta[0], gradient_prev[0]);
				break;
			default:
				break;
		}
	}

} /* namespace ml */

