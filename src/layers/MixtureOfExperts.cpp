/*
 * MixtureOfExperts.cpp
 *
 *  Created on: Feb 6, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/MixtureOfExperts.hpp>
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
	MixtureOfExperts::MixtureOfExperts(int experts, int neurons, std::string activation) :
			Layer(activation),
			m_experts(neurons),
			m_neurons(neurons),
			m_use_bias(true)
	{
	}

	MixtureOfExperts& MixtureOfExperts::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool MixtureOfExperts::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void MixtureOfExperts::setInputShape(const std::vector<Shape> &shapes)
	{
		const int first_dim = shapes[0].firstDim();
		const int last_dim = shapes[0].volumeWithoutFirstDim();
		m_input_shapes = { Shape( { first_dim, last_dim }) };
	}
	Shape MixtureOfExperts::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return Shape( { getInputShape().firstDim(), m_neurons });
	}
	Shape MixtureOfExperts::getWeightShape() const
	{
		return Shape( { m_neurons, getInputShape().lastDim() });
	}
	Shape MixtureOfExperts::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_neurons });
		else
			return Shape();
	}

	std::string MixtureOfExperts::name() const
	{
		return "MixtureOfExperts";
	}
	Json MixtureOfExperts::getConfig() const
	{
		Json result = Layer::getConfig();
		result["neurons"] = m_neurons;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> MixtureOfExperts::clone(const Json &config) const
	{
		std::unique_ptr<MixtureOfExperts> result = std::make_unique<MixtureOfExperts>(config["neurons"], config["nonlinearity"]);
		result->m_use_bias = config["use_bias"].getBool();
		result->loadConfig(config);
		return result;
	}

	void MixtureOfExperts::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		const Tensor flattened_input = flatten_input_tensor(input[0]);
		if (input.size() == 1)
			gemm_ex(context(), output, 1.0f, 'n', flattened_input, 't', getWeights().getParam(), 0.0f, output, getBias().getParam(), m_activation);
		else
		{
			const Tensor flattened_ext = flatten_input_tensor(input[1]);
			gemm_ex(context(), output, 1.0f, 'n', flattened_input, 't', getWeights().getParam(), 0.0f, flattened_ext, getBias().getParam(),
					m_activation);
		}
	}
	void MixtureOfExperts::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		Tensor empty;
		fusedBiasActCopyBackward(context(), gradient_next, output, 0.0f, empty, 0.0f, getBias().getGradient(), m_activation);

		Tensor tmp_grad = flatten_input_tensor(gradient_prev[0]);
		gemm(context(), 'n', 'n', tmp_grad, gradient_next, getWeights().getParam(), 1.0f, beta[0]);
		gemm(context(), 't', 'n', getWeights().getGradient(), gradient_next, flatten_input_tensor(input[0]), 1.0f, 0.0f);
	}

} /* namespace ml */

