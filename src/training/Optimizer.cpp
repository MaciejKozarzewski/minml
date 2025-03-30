/*
 * Optimizer.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/Optimizer.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>

#include <algorithm>
#include <cmath>

namespace
{
	bool is_fp16(const std::vector<ml::Tensor> &v)
	{
		return v.empty() ? false : (v[0].dtype() == ml::DataType::FLOAT16);
	}
}

namespace ml
{
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
		if (is_fp16(weights) and m_fp32_weights.size() != weights.size())
		{
			m_fp32_weights.clear();
			for (size_t i = 0; i < weights.size(); i++)
			{
				m_fp32_weights.emplace_back(weights[i].shape(), DataType::FLOAT32, weights[i].device());
				convertTensor(context, m_fp32_weights.back(), weights[i]);
			}
		}
		if (m_momentums.size() != weights.size())
		{
			m_momentums.clear();
			for (size_t i = 0; i < weights.size(); i++)
				m_momentums.emplace_back(weights[i].shape(), DataType::FLOAT32, weights[i].device());
		}
		if (m_variances.size() != weights.size())
		{
			m_variances.clear();
			for (size_t i = 0; i < weights.size(); i++)
				m_variances.emplace_back(weights[i].shape(), DataType::FLOAT32, weights[i].device());
		}

		m_steps++;
		if (is_fp16(weights))
			radamOptimize(context, scale, gradients, m_fp32_weights, m_momentums, m_variances, weights, m_learning_rate, m_beta1, m_beta2, m_steps);
		else
		{
			std::vector<Tensor> empty;
			radamOptimize(context, scale, gradients, weights, m_momentums, m_variances, empty, m_learning_rate, m_beta1, m_beta2, m_steps);
		}
	}

	Json RAdam::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "RAdam";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["steps"] = m_steps;
		result["fp32_weights"] = Json::array();
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			result["fp32_weights"][i] = m_fp32_weights[i].serialize(binary_data);
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
		m_fp32_weights = std::vector<Tensor>(json["fp32_weights"].size());
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			m_fp32_weights[i].unserialize(json["fp32_weights"][i], binary_data);
		m_momentums = std::vector<Tensor>(json["momentum"].size());
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].unserialize(json["momentum"][i], binary_data);
		m_variances = std::vector<Tensor>(json["variance"].size());
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].unserialize(json["variance"][i], binary_data);
	}

} /* namespace ml */
