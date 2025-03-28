/*
 * LearnableGlobalPooling.cpp
 *
 *  Created on: Mar 7, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/LearnableGlobalPooling.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/time_util.hpp>

namespace ml
{
	LearnableGlobalPooling::LearnableGlobalPooling(int expansionRatio, std::string activation) :
			Layer(activation),
			m_expansion_ratio(expansionRatio)
	{
	}

	void LearnableGlobalPooling::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalPooling layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape LearnableGlobalPooling::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().firstDim();
		const int channels = getInputShape().lastDim();
		return Shape( { batch_size, channels * m_expansion_ratio });
	}
	Shape LearnableGlobalPooling::getWeightShape() const
	{
		const int height = getInputShape().dim(1);
		const int width = getInputShape().dim(2);
		return Shape( { m_expansion_ratio, height * width });
	}
	Shape LearnableGlobalPooling::getBiasShape() const
	{
		return Shape( { m_expansion_ratio });
	}

	std::string LearnableGlobalPooling::name() const
	{
		return "LearnableGlobalPooling";
	}

	Json LearnableGlobalPooling::getConfig() const
	{
		Json result = Layer::getConfig();
		result["expansion_ratio"] = m_expansion_ratio;
		return result;
	}
	std::unique_ptr<Layer> LearnableGlobalPooling::clone(const Json &config) const
	{
		std::unique_ptr<LearnableGlobalPooling> result = std::make_unique<LearnableGlobalPooling>(config["expansion_ratio"].getInt());
		result->loadConfig(config);
		return result;
	}

	int LearnableGlobalPooling::getWorkspaceSize() const noexcept
	{
		if (isTrainable())
			return getWeightShape().volume() * getInputShape().firstDim();
		else
			return 0;
	}
	void LearnableGlobalPooling::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		const int batch_size = input[0].dim(0);
		const int height = input[0].dim(1);
		const int width = input[0].dim(2);
		const int channels = input[0].dim(3);

		Tensor tmp_in = input[0].view( { batch_size, height * width, channels });
		Tensor tmp_out = output.view( { batch_size, channels, m_expansion_ratio });

		gemmBatched(context(), 't', 't', tmp_out, tmp_in, getWeights().getParam(), 1, 0);
		activationForward(context(), 1.0f, output, 0.0f, output, m_activation);
	}
	void LearnableGlobalPooling::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		const int batch_size = input[0].dim(0);
		const int height = input[0].dim(1);
		const int width = input[0].dim(2);
		const int channels = input[0].dim(3);

		Tensor tmp_in = input[0].view( { batch_size, height * width, channels });
		Tensor tmp_prev = gradient_prev[0].view( { batch_size, height * width, channels });
		Tensor tmp_next = gradient_next.view( { batch_size, channels, m_expansion_ratio });
		Tensor tmp_update = m_workspace.lock()->view( { batch_size, m_expansion_ratio, height * width });

		activationBackward(context(), 1.0f, gradient_next, output, 0.0f, gradient_next, m_activation);

		gemmBatched(context(), 't', 't', tmp_prev, getWeights().getParam(), tmp_next, 1, beta[0]);
		gemmBatched(context(), 't', 't', tmp_update, tmp_next, tmp_in, 1, 0);

		sumOverFirstDim(context(), 1.0f, tmp_update.view( { batch_size, m_expansion_ratio * height * width }), 0.0f, getWeights().getParam());
	}

} /* namespace ml */

