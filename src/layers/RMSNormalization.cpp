/*
 * RMSNormalization.cpp
 *
 *  Created on: Jul 26, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/RMSNormalization.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/random.hpp>

namespace ml
{
	RMSNormalization::RMSNormalization(bool useGamma) :
			Layer()
	{
		m_use_gamma = useGamma;
	}

	RMSNormalization& RMSNormalization::useGamma(bool b) noexcept
	{
		m_use_gamma = b;
		return *this;
	}

	void RMSNormalization::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "LayerNormalization layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape RMSNormalization::getOutputShape() const
	{
		return getInputShape();
	}
	Shape RMSNormalization::getWeightShape() const
	{
		if (m_use_gamma)
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}

	std::string RMSNormalization::name() const
	{
		return "RMSNormalization";
	}
	Json RMSNormalization::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_gamma"] = m_use_gamma;
		return result;
	}

	std::unique_ptr<Layer> RMSNormalization::clone(const Json &config) const
	{
		std::unique_ptr<RMSNormalization> result = std::make_unique<RMSNormalization>(config["use_gamma"]);
		return result;
	}

	void RMSNormalization::init()
	{
		getWeights().getParam().setall(1.0f);
	}
	void RMSNormalization::setRegularizer(const Regularizer &regularizer)
	{
		getWeights().getRegularizer() = Regularizer(regularizer.getCoefficient(), 1.0f);
	}
	void RMSNormalization::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		rmsnormForward(context(), input[0], output, getWeights().getParam());
	}
	void RMSNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == input.size());

		rmsnormBackward(context(), input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient());
	}

} /* namespace ml */

