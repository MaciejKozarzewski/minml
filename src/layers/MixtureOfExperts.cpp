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
			m_experts(experts),
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
		const int experts = getInputShape().dim(0);
		if (experts != m_experts)
			throw LogicError(METHOD_NAME, "number of experts mismatch");
		m_input_shapes = shapes;
	}
	Shape MixtureOfExperts::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().dim(1);
		const int top_k = getInputShape().dim(2);
		return Shape( { m_experts, batch_size, top_k, m_neurons });
	}
	Shape MixtureOfExperts::getWeightShape() const
	{
		return Shape( { m_experts, m_neurons, getInputShape().lastDim() });
	}
	Shape MixtureOfExperts::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_experts, m_neurons });
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
		result["experts"] = m_experts;
		result["neurons"] = m_neurons;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> MixtureOfExperts::clone(const Json &config) const
	{
		std::unique_ptr<MixtureOfExperts> result = std::make_unique<MixtureOfExperts>(config["experts"].getInt(), config["neurons"].getInt(),
				config["nonlinearity"]);
		result->m_use_bias = config["use_bias"].getBool();
		result->loadConfig(config);
		return result;
	}

	void MixtureOfExperts::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const Tensor flattened_input = input[0].view().flatten( { 1, 2 });
		Tensor flattened_output = output.view().flatten( { 1, 2 });
		gemmBatched(context(), 'n', 't', flattened_output, flattened_input, getWeights().getParam(), 1.0f, 0.0f);
		addBiasAct(context(), 1.0f, flattened_output, getBias().getParam(), 0.0f, flattened_output, m_activation);
	}
	void MixtureOfExperts::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		const Tensor flattened_input = input[0].view().flatten( { 1, 2 });
		Tensor flattened_output = output.view().flatten( { 1, 2 });
		Tensor flattened_next = gradient_next.view().flatten( { 1, 2 });
		Tensor flattened_prev = gradient_prev[0].view().flatten( { 1, 2 });

		Tensor empty;
		fusedBiasActCopyBackward(context(), flattened_next, flattened_output, 0.0f, empty, 0.0f, getBias().getGradient(), m_activation);

		Tensor tmp_grad = flatten_input_tensor(gradient_prev[0]);
		gemmBatched(context(), 'n', 'n', flattened_prev, flattened_next, getWeights().getParam(), 1.0f, beta[0]);
		gemmBatched(context(), 't', 'n', getWeights().getGradient(), flattened_next, flattened_input, 1.0f, 0.0f);
	}

} /* namespace ml */


