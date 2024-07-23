/*
 * SqueezeAndExcitation.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/SqueezeAndExcitation.hpp>
#include <minml/utils/json.hpp>

#include <minml/utils/time_util.hpp>

namespace ml
{
	SqueezeAndExcitation::SqueezeAndExcitation(std::string activation) :
			Layer(activation)
	{
	}

	void SqueezeAndExcitation::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "SqueezeAndExcitation layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape SqueezeAndExcitation::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape SqueezeAndExcitation::getWeightShape() const
	{
		const int channels = getInputShape().lastDim();
		return Shape( { channels, 2 * channels });
	}

	std::string SqueezeAndExcitation::name() const
	{
		return "SqueezeAndExcitation";
	}

	int SqueezeAndExcitation::getWorkspaceSize() const noexcept
	{
		const int batch_size = getInputShape().firstDim();
		const int channels = getInputShape().lastDim();
		return 2 * (batch_size * 2 * channels) + batch_size * channels;
	}
	std::unique_ptr<Layer> SqueezeAndExcitation::clone(const Json &config) const
	{
		std::unique_ptr<SqueezeAndExcitation> result = std::make_unique<SqueezeAndExcitation>(config["nonlinearity"]);
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void SqueezeAndExcitation::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const int batch_size = input[0].firstDim();
		const int channels = input[0].lastDim();

		Tensor avg_max = m_workspace.lock()->view(Shape( { batch_size, 2, channels }));
		Tensor shifts = m_workspace.lock()->view(Shape( { batch_size, channels }), avg_max.volume());

		globalAvgAndMaxPoolingForward(context(), input[0], avg_max);

		avg_max.reshape( { batch_size, 2 * channels });
		gemm(context(), 'n', 't', shifts, avg_max, getWeights().getParam(), 1.0f, 0.0f);

		globalBroadcastingForward(context(), input[0], output, shifts, m_activation);
	}
	void SqueezeAndExcitation::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);

		const int batch_size = input[0].firstDim();
		const int channels = input[0].lastDim();

		Tensor avg_max = m_workspace.lock()->view(Shape( { batch_size, 2, channels }));
		globalAvgAndMaxPoolingForward(context(), input[0], avg_max);

		Tensor grad_shifts = m_workspace.lock()->view(Shape( { batch_size, channels }), avg_max.volume());
		globalBroadcastingBackward(context(), grad_shifts, gradient_next, output, m_activation);

		Tensor grad_avg_max = m_workspace.lock()->view(Shape( { batch_size, 2 * channels }), avg_max.volume() + grad_shifts.volume());
		gemm(context(), 'n', 'n', grad_avg_max, grad_shifts, getWeights().getParam(), 1.0f, 0.0f);

		grad_avg_max.reshape( { batch_size, 2, channels });
		globalAvgAndMaxPoolingBackward(context(), gradient_prev[0], grad_avg_max, input[0], avg_max);
		addTensors(context(), gradient_prev[0], gradient_prev[0], gradient_next);

		avg_max.reshape( { batch_size, 2 * channels });
		gemm(context(), 't', 'n', getWeights().getGradient(), grad_shifts, avg_max, 1.0f, 0.0f);
	}

} /* namespace ml */

