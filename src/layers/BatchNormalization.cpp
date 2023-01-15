/*
 * BatchNormalization.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/BatchNormalization.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	BatchNormalization::BatchNormalization(const std::string &activation, bool useGamma, bool useBeta, int historySize) :
			Layer(std::string(activation))
	{
		m_use_gamma = useGamma;
		m_use_beta = useBeta;
		m_history_size = historySize;
	}

	BatchNormalization& BatchNormalization::useGamma(bool b) noexcept
	{
		m_use_gamma = b;
		return *this;
	}
	BatchNormalization& BatchNormalization::useBeta(bool b) noexcept
	{
		m_use_beta = b;
		return *this;
	}

	void BatchNormalization::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "BatchNormalization layer expects single input shape");
		if (m_running_stats == nullptr)
			m_running_stats = std::make_unique<Tensor>(Shape( { m_history_size, 2 * shapes[0].lastDim() }), dtype(), device());
		m_input_shapes = shapes;
	}
	Shape BatchNormalization::getOutputShape() const
	{
		return getInputShape();
	}
	Shape BatchNormalization::getWeightShape() const
	{
		return Shape( { 4, getInputShape().lastDim() });
	}
	Shape BatchNormalization::getBiasShape() const
	{
		return Shape();
	}

	std::string BatchNormalization::name() const
	{
		return "BatchNormalization";
	}
	Json BatchNormalization::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_gamma"] = m_use_gamma;
		result["use_beta"] = m_use_beta;
		result["history_size"] = m_history_size;
		return result;
	}

	void BatchNormalization::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		if (m_running_stats != nullptr)
			m_running_stats->moveTo(device());
	}

	std::unique_ptr<Layer> BatchNormalization::clone(const Json &config) const
	{
		std::unique_ptr<BatchNormalization> result = std::make_unique<BatchNormalization>(config["nonlinearity"], config["use_gamma"],
				config["use_beta"], config["history_size"]);
		return result;
	}

	void BatchNormalization::init()
	{
		getWeights().getParam().setall(context(), 1.0f);
		getWeights().getParam().view(Shape( { getInputShape().lastDim() }), 3 * getInputShape().lastDim()).setall(context(), 0.0f);
		m_running_stats->zeroall(context());
		m_total_steps = 0;
		m_running_id = 0;
	}
	void BatchNormalization::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		if (input[0].shape().firstDim() == 1)
			batchnormInference(context(), input[0], output, getWeights().getParam(), m_activation);
		else
			batchnormForward(context(), input[0], output, getWeights().getParam(), *m_running_stats, m_running_id, m_activation);
	}
	void BatchNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		batchnormBackward(context(), input[0], output, gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient(),
				*m_running_stats, m_running_id, m_activation);
	}
	void BatchNormalization::learn()
	{
		Layer::learn();
		m_total_steps++;
		m_running_id = (m_running_id + 1) % m_history_size;
		batchnormUpdate(context(), *m_running_stats, std::min(m_history_size, m_total_steps), getWeights().getParam(), m_use_gamma, m_use_beta);
	}

} /* namespace ml */

