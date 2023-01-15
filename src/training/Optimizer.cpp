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
	Optimizer::Optimizer(float learningRate, float beta1, float beta2) :
			m_learning_rate(learningRate),
			m_beta1(beta1),
			m_beta2(beta2)
	{
	}
	Optimizer::Optimizer(const Optimizer &other) :
			m_learning_rate(other.m_learning_rate),
			m_beta1(other.m_beta1),
			m_beta2(other.m_beta2),
			m_steps(other.m_steps)
	{
		if (other.m_momentum != nullptr)
			this->m_momentum = std::make_unique<Tensor>(*other.m_momentum);
		if (other.m_variance != nullptr)
			this->m_variance = std::make_unique<Tensor>(*other.m_variance);
	}
	Optimizer& Optimizer::operator=(const Optimizer &other)
	{
		this->m_learning_rate = other.m_learning_rate;
		this->m_beta1 = other.m_beta1;
		this->m_beta2 = other.m_beta2;
		this->m_steps = other.m_steps;
		if (other.m_momentum != nullptr)
			this->m_momentum = std::make_unique<Tensor>(*other.m_momentum);
		if (other.m_variance != nullptr)
			this->m_variance = std::make_unique<Tensor>(*other.m_variance);
		return *this;
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
		if (m_momentum != nullptr)
			m_momentum->zeroall(context);
		if (m_variance != nullptr)
			m_variance->zeroall(context);
	}
	void Optimizer::moveTo(Device newDevice)
	{
		if (m_momentum != nullptr)
			m_momentum->moveTo(newDevice);
		if (m_variance != nullptr)
			m_variance->moveTo(newDevice);
	}
	void Optimizer::apply(const Context &context, Parameter &param)
	{
		size_t length = param.getParam().volume();
		if (length == 0)
			return;

		if (m_momentum == nullptr)
			m_momentum = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());
		if (m_variance == nullptr)
			m_variance = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());

		m_steps++;
		float learning_rate = m_learning_rate;
		if (m_steps < 10000)
			learning_rate *= sqrt(1.0f - pow(m_beta2, m_steps)) / (1.0f - pow(m_beta1, m_steps));

		adamOptimize(context, param.getParam(), param.getGradient(), *m_momentum, *m_variance, learning_rate, m_beta1, m_beta2);
	}

	Json Optimizer::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "ADAM";
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["steps"] = m_steps;
		result["momentum"] = (m_momentum == nullptr) ? Json() : m_momentum->serialize(binary_data);
		result["variance"] = (m_variance == nullptr) ? Json() : m_variance->serialize(binary_data);
		return result;
	}
	void Optimizer::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "ADAM");
		m_learning_rate = json["learning rate"];
		m_beta1 = json["beta1"];
		m_beta2 = json["beta2"];
		m_steps = json["steps"];
		m_momentum = json["momentum"].isNull() ? nullptr : std::make_unique<Tensor>(json["momentum"], binary_data);
		m_variance = json["variance"].isNull() ? nullptr : std::make_unique<Tensor>(json["variance"], binary_data);
	}

} /* namespace ml */
