/*
 * Regularizer.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/Regularizer.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/random.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>

#include <cmath>

namespace ml
{
	Regularizer::Regularizer(float coefficient, float offset) :
			m_coefficient(coefficient),
			m_offset(offset)
	{
	}
	float Regularizer::getCoefficient() const noexcept
	{
		return m_coefficient;
	}
	float Regularizer::getOffset() const noexcept
	{
		return m_offset;
	}
	void Regularizer::apply(const Context &context, Parameter &param)
	{
		l2Regularization(context, param.getGradient(), param.getParam(), m_coefficient, m_offset);
	}

	Json Regularizer::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "RegularizerL2";
		result["coefficient"] = m_coefficient;
		result["offset"] = m_offset;
		return result;
	}
	void Regularizer::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "RegularizerL2");
		m_coefficient = json["coefficient"];
		m_offset = json["offset"];
	}

	RegularizerL2::RegularizerL2(float scale) :
			m_scale(scale)
	{
	}
	float RegularizerL2::getScale() const noexcept
	{
		return m_scale;
	}
	void RegularizerL2::apply(const Context &context, float alpha, const std::vector<Tensor> &weights, std::vector<Tensor> &gradients)
	{
		l2Regularization(context, gradients, weights, alpha * m_scale);
	}

	Json RegularizerL2::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = "RegularizerL2";
		result["scale"] = m_scale;
		return result;
	}
	void RegularizerL2::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "RegularizerL2");
		m_scale = json["scale"];
	}
} /* namespace ml */

