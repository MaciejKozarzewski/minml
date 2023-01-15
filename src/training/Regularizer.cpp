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
} /* namespace ml */

