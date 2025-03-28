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
	LayerNormalization::LayerNormalization(bool useGamma, bool useBeta) :
			Layer()
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
		return Shape( { getInputShape().lastDim() });
	}
	Shape LayerNormalization::getBiasShape() const
	{
		return Shape( { getInputShape().lastDim() });
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

		Tensor ext;
		if (input.size() == 2)
			ext = input.at(1).view();

		layernormForward(context(), input[0], output, getWeights().getParam(), getBias().getParam(), ext);
	}
	void LayerNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == input.size());

		layernormBackward(context(), input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient(),
				getBias().getGradient(), beta[0]);
	}

} /* namespace ml */

