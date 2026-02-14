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
			m_use_bias(true)
	{
	}

	Dense& Dense::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool Dense::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Dense::setInputShape(const std::vector<Shape> &shapes)
	{
		const int first_dim = shapes[0].firstDim();
		const int last_dim = shapes[0].volumeWithoutFirstDim();
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
		return Shape( { m_neurons, getInputShape().lastDim() });
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
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> Dense::clone(const Json &config) const
	{
		std::unique_ptr<Dense> result = std::make_unique<Dense>(config["neurons"], config["nonlinearity"]);
		result->m_use_bias = config["use_bias"].getBool();
		result->loadConfig(config);
		return result;
	}

	void Dense::forward(const std::vector<Tensor> &input, Tensor &output)
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
	void Dense::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		Tensor empty;
		Tensor flattened_dy = gradient_next.view().flatten( { 0, 1, 2 });
		const Tensor flattened_y = output.view().flatten( { 0, 1, 2 });
		fusedBiasActCopyBackward(context(), flattened_dy, flattened_y, 0.0f, empty, 0.0f, getBias().getGradient(), m_activation);

		Tensor tmp_grad = flatten_input_tensor(gradient_prev[0]);
		gemm(context(), 'n', 'n', tmp_grad, gradient_next, getWeights().getParam(), 1.0f, beta[0]);
		gemm(context(), 't', 'n', getWeights().getGradient(), gradient_next, flatten_input_tensor(input[0]), 1.0f, 0.0f);
	}

} /* namespace ml */

