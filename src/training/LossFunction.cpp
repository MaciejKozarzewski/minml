/*
 * LossFunction.cpp
 *
 *  Created on: May 23, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/LossFunction.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	CrossEntropyLoss::CrossEntropyLoss(float weight) :
			m_weight(weight)
	{
	}
	float CrossEntropyLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return m_weight * crossEntropyLoss(context, output, target);
	}
	void CrossEntropyLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		crossEntropyGradient(context, gradient, output, target, m_weight);
	}
	Json CrossEntropyLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", "CrossEntropyLoss" }, { "weight", m_weight } });
	}
	void CrossEntropyLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "CrossEntropyLoss");
		m_weight = json["weight"].getDouble();
	}
	std::unique_ptr<LossFunction> CrossEntropyLoss::clone() const
	{
		return std::make_unique<CrossEntropyLoss>(m_weight);
	}

	MeanSquaredLoss::MeanSquaredLoss(float weight) :
			m_weight(weight)
	{
	}
	float MeanSquaredLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return m_weight * meanSquaredLoss(context, output, target);
	}
	void MeanSquaredLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		meanSquaredGradient(context, gradient, output, target, m_weight);
	}
	Json MeanSquaredLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", "MeanSquaredLoss" }, { "weight", m_weight } });
	}
	void MeanSquaredLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "MeanSquaredLoss");
		m_weight = json["weight"].getDouble();
	}
	std::unique_ptr<LossFunction> MeanSquaredLoss::clone() const
	{
		return std::make_unique<MeanSquaredLoss>(m_weight);
	}

	ValueHeadLoss::ValueHeadLoss(float weight) :
			m_weight(weight)
	{
	}
	float ValueHeadLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return m_weight * valueHeadLoss(context, output, target);
	}
	void ValueHeadLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		valueHeadGradient(context, gradient, output, target, m_weight);
	}
	Json ValueHeadLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", "ValueHeadLoss" }, { "weight", m_weight } });
	}
	void ValueHeadLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "ValueHeadLoss");
		m_weight = json["weight"].getDouble();
	}
	std::unique_ptr<LossFunction> ValueHeadLoss::clone() const
	{
		return std::make_unique<ValueHeadLoss>(m_weight);
	}

} /* namespace ml */

