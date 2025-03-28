/*
 * Optimizer.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/Optimizer.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/utils/json.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>

#include <algorithm>
#include <cmath>

namespace ml
{
	Optimizer::Optimizer(float learningRate, float beta1, float beta2, float weight_decay) :
			m_learning_rate(learningRate),
			m_beta1(beta1),
			m_beta2(beta2),
			m_weight_decay(weight_decay)
	{
	}

	float Optimizer::getLearningRate() const noexcept
	{
		return m_learning_rate;
	}
	void Optimizer::setLearningRate(float lr) noexcept
	{
		this->m_learning_rate = lr;
	}
	int Optimizer::getSteps() const noexcept
	{
		return m_steps;
	}

	void Optimizer::restart(const Context &context) noexcept
	{
		m_steps = 0;
		m_momentum.zeroall();
		m_variance.zeroall();
	}
	void Optimizer::moveTo(Device newDevice)
	{
		m_momentum.moveTo(newDevice);
		m_variance.moveTo(newDevice);
	}
	void Optimizer::apply(const Context &context, Parameter &param, float scale)
	{
		if (param.getParam().isEmpty())
			return;

		if (m_momentum.isEmpty())
			m_momentum = zeros_like(param.getParam());
		if (m_variance.isEmpty())
			m_variance = zeros_like(param.getParam());

		m_steps++;
		radamOptimize(context, scale, param.getGradient(), param.getParam(), m_momentum, m_variance, m_learning_rate, m_beta1, m_beta2, m_steps);
		param.getGradient().zeroall(context);
	}

	Json Optimizer::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "ADAM";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["steps"] = m_steps;
		result["momentum"] = m_momentum.serialize(binary_data);
		result["variance"] = m_variance.serialize(binary_data);
		return result;
	}
	void Optimizer::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "ADAM");
		m_learning_rate = json["learning rate"];
		m_beta1 = json["beta1"];
		m_beta2 = json["beta2"];
		m_steps = json["steps"];
		m_momentum = json["momentum"].isNull() ? Tensor() : Tensor(json["momentum"], binary_data);
		m_variance = json["variance"].isNull() ? Tensor() : Tensor(json["variance"], binary_data);
	}

	RAdam::RAdam(float learningRate, float beta1, float beta2) :
			m_learning_rate(learningRate),
			m_beta1(beta1),
			m_beta2(beta2)
	{
	}

	float RAdam::getLearningRate() const noexcept
	{
		return m_learning_rate;
	}
	void RAdam::setLearningRate(float lr) noexcept
	{
		this->m_learning_rate = lr;
	}
	int RAdam::getSteps() const noexcept
	{
		return m_steps;
	}

	void RAdam::restart(const Context &context) noexcept
	{
		m_steps = 0;
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].zeroall();
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].zeroall();
	}
	void RAdam::moveTo(Device newDevice)
	{
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].moveTo(newDevice);
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].moveTo(newDevice);
	}
	void RAdam::apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale)
	{
		if (m_momentums.size() != weights.size())
		{
			m_momentums.clear();
			for (size_t i = 0; i < weights.size(); i++)
				m_momentums.push_back(zeros_like(weights[i]));
		}
		if (m_variances.size() != weights.size())
		{
			m_variances.clear();
			for (size_t i = 0; i < weights.size(); i++)
				m_variances.push_back(zeros_like(weights[i]));
		}

		m_steps++;
		radamOptimize(context, scale, gradients, weights, m_momentums, m_variances, m_learning_rate, m_beta1, m_beta2, m_steps);
	}

	Json RAdam::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "RAdam";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["steps"] = m_steps;
		result["momentums"] = Json::array();
		for (size_t i = 0; i < m_momentums.size(); i++)
			result["momentums"][i] = m_momentums[i].serialize(binary_data);
		result["variances"] = Json::array();
		for (size_t i = 0; i < m_variances.size(); i++)
			result["variances"][i] = m_variances[i].serialize(binary_data);
		return result;
	}
	void RAdam::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "RAdam");
		m_learning_rate = json["learning rate"];
		m_beta1 = json["beta1"];
		m_beta2 = json["beta2"];
		m_steps = json["steps"];
		m_momentums = std::vector<Tensor>(json["momentum"].size());
		m_variances = std::vector<Tensor>(json["variance"].size());
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].unserialize(json["momentum"][i], binary_data);
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].unserialize(json["variance"][i], binary_data);
	}

} /* namespace ml */
