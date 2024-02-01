/*
 * GlobalPooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/GlobalPooling.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

#include <minml/utils/time_util.hpp>

namespace ml
{
	GlobalPooling::GlobalPooling(std::string activation) :
			Layer(activation)
	{
	}

	void GlobalPooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape GlobalPooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape GlobalPooling::getWeightShape() const
	{
		const int channels = getInputShape().lastDim();
		return Shape( { channels, 2 * channels });
	}

	std::string GlobalPooling::name() const
	{
		return "GlobalPooling";
	}

	int GlobalPooling::getWorkspaceSize() const noexcept
	{
		const int batch_size = getInputShape().firstDim();
		const int channels = getInputShape().lastDim();
		return 2 * (batch_size * 2 * channels) + batch_size * channels;
	}
	std::unique_ptr<Layer> GlobalPooling::clone(const Json &config) const
	{
		std::unique_ptr<GlobalPooling> result = std::make_unique<GlobalPooling>(config["nonlinearity"]);
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void GlobalPooling::forward(const std::vector<Tensor> &input, Tensor &output)
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
	void GlobalPooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
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

