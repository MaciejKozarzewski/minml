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
#include <minml/utils/random.hpp>

namespace
{
	using namespace ml;
	Tensor get_statistics(Tensor &running_stats, int id)
	{
		return running_stats.view( { running_stats.lastDim() }, { id, 0 });
	}
}

namespace ml
{
	BatchNormalization::BatchNormalization(std::string activation, bool useGamma, bool useBeta, int historySize) :
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
	BatchNormalization& BatchNormalization::historySize(int s) noexcept
	{
		if (m_history_size != s)
			m_historical_stats = Tensor();
		m_history_size = s;
		return *this;
	}
	bool BatchNormalization::isUsingGamma() const noexcept
	{
		return m_use_gamma;
	}
	bool BatchNormalization::isUsingBeta() const noexcept
	{
		return m_use_beta;
	}

	void BatchNormalization::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "BatchNormalization layer expects single input shape");
		if (m_historical_stats.isEmpty() and isTrainable())
			m_historical_stats = Tensor(Shape( { m_history_size, 3 * shapes[0].lastDim() }), DataType::FLOAT32, device());
		if (m_avg_var.isEmpty())
			m_avg_var = Tensor(Shape( { 2, shapes[0].lastDim() }), DataType::FLOAT32, device());
		m_input_shapes = shapes;
	}
	Shape BatchNormalization::getOutputShape() const
	{
		return getInputShape();
	}
	Shape BatchNormalization::getWeightShape() const
	{
		if (isUsingGamma())
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}
	Shape BatchNormalization::getBiasShape() const
	{
		if (isUsingBeta())
			return Shape( { getInputShape().lastDim() });
		else
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
		result["total_steps"] = m_total_steps;
		result["running_id"] = m_history_id;
		return result;
	}
	Json BatchNormalization::saveParameters(SerializedObject &binary_data) const
	{
		Json result = Layer::saveParameters(binary_data);
		result["running_stats"] = m_historical_stats.serialize(binary_data);
		result["avg_var"] = m_avg_var.serialize(binary_data);
		return result;
	}
	void BatchNormalization::loadParameters(const Json &json, const SerializedObject &binary_data)
	{
		Layer::loadParameters(json, binary_data);
		if (json.hasKey("running_stats") and not json["running_stats"].isNull())
			m_historical_stats.unserialize(json["running_stats"], binary_data);
		if (json.hasKey("avg_var") and not json["avg_var"].isNull())
			m_avg_var.unserialize(json["avg_var"], binary_data);
	}

	void BatchNormalization::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_historical_stats.moveTo(device());
		m_avg_var.moveTo(device());
	}

	std::unique_ptr<Layer> BatchNormalization::clone(const Json &config) const
	{
		std::unique_ptr<BatchNormalization> result = std::make_unique<BatchNormalization>(config["nonlinearity"], config["use_gamma"],
				config["use_beta"], config["history_size"]);
		result->loadConfig(config);
		if (config.hasKey("total_steps"))
			result->m_total_steps = config["total_steps"];
		if (config.hasKey("running_id"))
			result->m_history_id = config["running_id"];
		return result;
	}

	void BatchNormalization::init()
	{
		const int last_dim = getInputShape().lastDim();
		m_avg_var.view(Shape( { last_dim }), { 0, 0 }).setall(context(), 0.0f);
		m_avg_var.view(Shape( { last_dim }), { 1, 0 }).setall(context(), 1.0f);

		getWeights().getParam().setall(context(), 1.0f);
		getBias().getParam().setall(context(), 0.0f);

		m_historical_stats.zeroall(context());
		m_total_steps = 0;
		m_history_id = 0;
	}
	void BatchNormalization::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		if (isTrainable())
		{
			if (input[0].shape().volumeWithoutLastDim() == 1)
				throw LogicError(METHOD_NAME, "cannot calculate batch normalization on tensor of shape " + input[0].shape().toString());

			Tensor stats = get_statistics(m_historical_stats, m_history_id);
			batchnormForward(context(), 1.0f, input[0], getWeights().getParam(), getBias().getParam(), 0.0f, output, stats, m_activation);
		}
		else
			batchnormInference(context(), 1.0f, input[0], getWeights().getParam(), getBias().getParam(), m_avg_var, 0.0f, output, m_activation);
	}
	void BatchNormalization::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		const Tensor stats = get_statistics(m_historical_stats, m_history_id);
		batchnormBackward(context(), 1.0f, input[0], output, gradient_next, getWeights().getParam(), getBias().getParam(), beta[0], gradient_prev[0],
				0.0f, getWeights().getGradient(), getBias().getGradient(), stats, m_activation);
	}
	void BatchNormalization::updateStatistics()
	{
		m_total_steps++;
		m_history_id = (m_history_id + 1) % m_history_size;
		const Tensor stats = m_historical_stats.view( { std::min(m_history_size, m_total_steps), m_historical_stats.lastDim() });
		batchnormUpdate(context(), stats, m_avg_var);
	}
	Tensor& BatchNormalization::getStatistics()
	{
		return m_avg_var;
	}

} /* namespace ml */

