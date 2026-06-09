/*
 * LayerNormalization.cpp
 *
 *  Created on: May 7, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/LayerNormalization.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/random.hpp>

namespace ml
{
	LayerNormalization::LayerNormalization(std::string activation, bool useGamma, bool useBeta) :
			Layer(activation)
	{
		m_use_gamma = useGamma;
		m_use_beta = useBeta;
	}

	LayerNormalization& LayerNormalization::useGamma(bool b) noexcept
	{
		m_use_gamma = b;
		return *this;
	}
	LayerNormalization& LayerNormalization::useBeta(bool b) noexcept
	{
		m_use_beta = b;
		return *this;
	}

	void LayerNormalization::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "LayerNormalization layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape LayerNormalization::getOutputShape() const
	{
		return getInputShape();
	}
	Shape LayerNormalization::getWeightShape() const
	{
		if (m_use_gamma)
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}
	Shape LayerNormalization::getBiasShape() const
	{
		if (m_use_beta)
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}

	std::string LayerNormalization::name() const
	{
		return "LayerNormalization";
	}
	Json LayerNormalization::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_gamma"] = m_use_gamma;
		result["use_beta"] = m_use_beta;
		return result;
	}

	std::unique_ptr<Layer> LayerNormalization::clone(const Json &config) const
	{
		std::unique_ptr<LayerNormalization> result = std::make_unique<LayerNormalization>(config["use_gamma"], config["use_beta"]);
		result->loadConfig(config);
		return result;
	}

	void LayerNormalization::init()
	{
		getWeights().getParam().setall(1.0f);
		getBias().getParam().setall(0.0f);
	}
	void LayerNormalization::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() <= 2);

		float beta = 0.0f;
		if (input.size() == 2)
		{
			output.copyFrom(context(), input[1]);
			beta = 1.0f;
		}

		layernormForward(context(), 1.0f, input[0], beta, output, getWeights().getParam(), getBias().getParam(), m_activation);
	}
	void LayerNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		assert(gradient_prev.size() == input.size());

		if (input.size() == 2)
		{
			Tensor empty;
			fusedBiasActCopyBackward(context(), gradient_next, output, beta[1], gradient_prev[1], 0.0f, empty, ActivationType::LINEAR);
		}
		activationBackward(context(), 1.0f, gradient_next, output, 0.0f, gradient_next, m_activation);

		layernormBackward(context(), 1.0f, input[0], beta[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient(),
				getBias().getGradient(), 1.0f);
	}

} /* namespace ml */

