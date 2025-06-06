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
	bool check_same_device(const std::vector<ml::Tensor> &lhs, const std::vector<ml::Tensor> &rhs)
	{
		if (lhs.empty() or rhs.empty())
			return true;
		ml::Device device = lhs[0].device();
		for (size_t i = 0; i < lhs.size(); i++)
			if (lhs[i].device() != device or rhs[i].device() != device)
				return false;
		return true;
	}
}

namespace ml
{
	RAdam::RAdam(float learningRate, float beta1, float beta2, float weightDecay) :
			m_learning_rate(learningRate),
			m_beta1(beta1),
			m_beta2(beta2),
			m_weight_decay(weightDecay)
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
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			m_fp32_weights[i].moveTo(newDevice);
	}
	void RAdam::convertTo(const Context &context, DataType newType)
	{
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].convertTo(context, newType);
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].convertTo(context, newType);
	}
	void RAdam::apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale)
	{
		assert(check_same_device(weights, gradients));
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
		assert(check_same_device(weights, m_fp32_weights));
		assert(check_same_device(weights, m_momentums));
		assert(check_same_device(weights, m_variances));

		m_steps++;
		if (is_fp16(weights))
			radamOptimize(context, scale, gradients, m_fp32_weights, m_momentums, m_variances, weights, m_learning_rate, m_beta1, m_beta2, m_steps,
					m_weight_decay);
		else
		{
			std::vector<Tensor> empty;
			radamOptimize(context, scale, gradients, weights, m_momentums, m_variances, empty, m_learning_rate, m_beta1, m_beta2, m_steps,
					m_weight_decay);
		}
	}

	Json RAdam::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "RAdam";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["weight_decay"] = m_weight_decay;
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
		m_weight_decay = json["weight_decay"];
		m_steps = json["steps"];

		m_fp32_weights = std::vector<Tensor>(json["fp32_weights"].size());
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			m_fp32_weights[i].unserialize(json["fp32_weights"][i], binary_data);

		m_momentums = std::vector<Tensor>(json["momentums"].size());
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].unserialize(json["momentums"][i], binary_data);

		m_variances = std::vector<Tensor>(json["variances"].size());
		for (size_t i = 0; i < m_variances.size(); i++)
			m_variances[i].unserialize(json["variances"][i], binary_data);
	}

	Lion::Lion(float learningRate, float beta1, float beta2, float weightDecay) :
			m_learning_rate(learningRate),
			m_beta1(beta1),
			m_beta2(beta2),
			m_weight_decay(weightDecay)
	{
	}

	float Lion::getLearningRate() const noexcept
	{
		return m_learning_rate;
	}
	void Lion::setLearningRate(float lr) noexcept
	{
		this->m_learning_rate = lr;
	}
	int Lion::getSteps() const noexcept
	{
		return m_steps;
	}

	void Lion::restart(const Context &context) noexcept
	{
		m_steps = 0;
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].zeroall();
	}
	void Lion::moveTo(Device newDevice)
	{
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].moveTo(newDevice);
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			m_fp32_weights[i].moveTo(newDevice);
	}
	void Lion::convertTo(const Context &context, DataType newType)
	{
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].convertTo(context, newType);
	}
	void Lion::apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale)
	{
		assert(check_same_device(weights, gradients));
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
		assert(check_same_device(weights, m_fp32_weights));
		assert(check_same_device(weights, m_momentums));

		m_steps++;
		if (is_fp16(weights))
			lionOptimize(context, scale, gradients, m_fp32_weights, m_momentums, weights, m_learning_rate, m_beta1, m_beta2, m_steps,
					m_weight_decay);
		else
		{
			std::vector<Tensor> empty;
			lionOptimize(context, scale, gradients, weights, m_momentums, empty, m_learning_rate, m_beta1, m_beta2, m_steps, m_weight_decay);
		}
	}

	Json Lion::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "Lion";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["weight_decay"] = m_weight_decay;
		result["steps"] = m_steps;

		result["fp32_weights"] = Json::array();
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			result["fp32_weights"][i] = m_fp32_weights[i].serialize(binary_data);

		result["momentums"] = Json::array();
		for (size_t i = 0; i < m_momentums.size(); i++)
			result["momentums"][i] = m_momentums[i].serialize(binary_data);

		return result;
	}
	void Lion::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "Lion");
		m_learning_rate = json["learning rate"];
		m_beta1 = json["beta1"];
		m_beta2 = json["beta2"];
		m_weight_decay = json["weight_decay"];
		m_steps = json["steps"];

		m_fp32_weights = std::vector<Tensor>(json["fp32_weights"].size());
		for (size_t i = 0; i < m_fp32_weights.size(); i++)
			m_fp32_weights[i].unserialize(json["fp32_weights"][i], binary_data);

		m_momentums = std::vector<Tensor>(json["momentums"].size());
		for (size_t i = 0; i < m_momentums.size(); i++)
			m_momentums[i].unserialize(json["momentums"][i], binary_data);
	}

} /* namespace ml */
