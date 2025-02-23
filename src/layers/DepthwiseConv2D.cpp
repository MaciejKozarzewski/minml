/*
 * DepthwiseConv2D.cpp
 *
 *  Created on: Jan 27, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/string_util.hpp>
#include <minml/utils/time_util.hpp>

#include <cmath>

namespace ml
{
	DepthwiseConv2D::DepthwiseConv2D(int filters, int kernelSize, bool useBias) :
			Layer("linear")
	{
		m_filters = filters;
		m_kernel_size = kernelSize;
		m_use_bias = useBias;
	}

	DepthwiseConv2D& DepthwiseConv2D::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool DepthwiseConv2D::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void DepthwiseConv2D::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "DepthwiseConv2D layer expects one input shape");
		if (shapes[0].rank() != 4)
			throw IllegalArgument(METHOD_NAME, "DepthwiseConv2D layer expects 4D shapes");

		m_input_shapes = shapes;
	}
	Shape DepthwiseConv2D::getOutputShape() const
	{
		return getInputShape();
	}
	Shape DepthwiseConv2D::getWeightShape() const
	{
		return Shape( { m_kernel_size, m_kernel_size, m_filters });
	}
	Shape DepthwiseConv2D::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_filters });
		else
			return Shape();
	}

	std::string DepthwiseConv2D::name() const
	{
		return "DepthwiseConv2D";
	}
	Json DepthwiseConv2D::getConfig() const
	{
		Json result = Layer::getConfig();
		result["filters"] = m_filters;
		result["kernel_size"] = m_kernel_size;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> DepthwiseConv2D::clone(const Json &config) const
	{
		std::unique_ptr<DepthwiseConv2D> result = std::make_unique<DepthwiseConv2D>(config["filters"], config["kernel_size"], config["use_bias"]);
		result->loadConfig(config);
		return result;
	}
	void DepthwiseConv2D::init()
	{
		m_initializer.init_weights(context(), getWeights(), 0.1f * std::sqrt(1.0f / m_filters), 0.0f);
		m_initializer.init_bias(context(), getBias(), 0.1f, 0.0f);
	}

	void DepthwiseConv2D::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		depthwiseConvForward(context(), input[0], getWeights().getParam(), output, getBias().getParam());
	}
	void DepthwiseConv2D::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1 && gradient_prev.size() == 1);
		depthwiseConvBackward(context(), gradient_next, getWeights().getParam(), gradient_prev[0]);
		depthwiseConvUpdate(context(), input[0], gradient_next, getWeights().getGradient());

		if (isUsingBias())
			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 0);
	}

} /* namespace ml */

