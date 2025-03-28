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

