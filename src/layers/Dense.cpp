/*
 * Dense.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Dense.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace
{
	ml::Tensor flatten_input_tensor(const ml::Tensor &t)
	{
		const int first_dim = t.firstDim();
		const int last_dim = t.shape().volumeWithoutFirstDim();
		return t.view(ml::Shape( { first_dim, last_dim }));
	}
}

namespace ml
{
	Dense::Dense(int neurons, std::string activation) :
			Layer(activation),
			m_neurons(neurons),
			m_use_weights(true),
			m_use_bias(true)
	{
	}

	Dense& Dense::useWeights(bool b) noexcept
	{
		if (b != m_use_weights)
			m_weights = nullptr;
		m_use_weights = b;
		return *this;
	}
	Dense& Dense::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool Dense::isUsingWeights() const noexcept
	{
		return m_use_weights;
	}
	bool Dense::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Dense::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Dense layer expects single input shape");
		const int first_dim = shapes[0].firstDim();
		const int last_dim = shapes[0].volumeWithoutFirstDim();
		if (not isUsingWeights() and m_neurons != last_dim)
			throw IllegalArgument(METHOD_NAME, "Dense layer without weights must have input and output of the same shape");

		m_input_shapes = { Shape( { first_dim, last_dim }) };
	}
	Shape Dense::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return Shape( { getInputShape().firstDim(), m_neurons });
	}
	Shape Dense::getWeightShape() const
	{
		if (m_use_weights)
			return Shape( { m_neurons, getInputShape().lastDim() });
		else
			return Shape();
	}
	Shape Dense::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_neurons });
		else
			return Shape();
	}

	std::string Dense::name() const
	{
		return "Dense";
	}
	Json Dense::getConfig() const
	{
		Json result = Layer::getConfig();
		result["neurons"] = m_neurons;
		result["use_weights"] = m_use_weights;
		result["use_bias"] = m_use_bias;
		return result;
	}

	int Dense::getWorkspaceSize() const noexcept
	{
		if (isUsingWeights())
			return getWeightShape().volume();
		else
			return 0;
	}
	std::unique_ptr<Layer> Dense::clone(const Json &config) const
	{
		std::unique_ptr<Dense> result = std::make_unique<Dense>(config["neurons"], config["nonlinearity"]);
		if (config.hasKey("use_weights"))
			result->m_use_weights = config["use_weights"].getBool();
		result->m_use_bias = config["use_bias"].getBool();
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void Dense::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		if (isUsingWeights())
		{
			if (device().isCPU())
			{
				const Tensor flattened_input = flatten_input_tensor(input[0]);
				const float beta = isUsingBias() ? 1.0f : 0.0f;
				if (m_activation == ActivationType::RELU or m_activation == ActivationType::LINEAR)
					gemm_ex(context(), output, 1.0f, 'n', flattened_input, 't', getWeights().getParam(), beta, getBias().getParam(),
							getBias().getParam(), m_activation);
				else
				{
					gemm_ex(context(), output, 1.0f, 'n', flattened_input, 't', getWeights().getParam(), beta, getBias().getParam(),
							getBias().getParam(), ActivationType::LINEAR);
					activationForward(context(), output, output, m_activation);
				}
			}
			else
			{
				gemm(context(), 'n', 't', output, flatten_input_tensor(input[0]), getWeights().getParam(), 1, 0);
				if (isUsingBias())
					addBiasAct(context(), output, output, getBias().getParam(), m_activation);
				else
					activationForward(context(), output, output, m_activation);
			}
		}
		else
		{
			output.copyFrom(context(), flatten_input_tensor(input[0]));
			if (isUsingBias())
				addBiasAct(context(), output, output, getBias().getParam(), m_activation);
			else
				activationForward(context(), output, output, m_activation);
		}
	}
	void Dense::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		activationBackward(context(), gradient_next, gradient_next, output, m_activation);
		if (isUsingWeights())
		{
			Tensor tmp_grad = flatten_input_tensor(gradient_prev[0]);
			gemm(context(), 'n', 'n', tmp_grad, gradient_next, getWeights().getParam(), 1, 0);
			gemm(context(), 't', 'n', getWeights().getGradient(), gradient_next, flatten_input_tensor(input[0]), 1, 0);
		}
		else
		{
			Tensor tmp = flatten_input_tensor(gradient_prev[0]);
			tmp.copyFrom(context(), gradient_next);
		}

		if (isUsingBias())
			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 0);
	}

} /* namespace ml */

