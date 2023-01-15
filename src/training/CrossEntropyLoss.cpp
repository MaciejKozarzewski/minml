/*
 * CrossEntropyLoss.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/CrossEntropyLoss.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	float CrossEntropyLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return crossEntropyLoss(context, output, target);
	}
	void CrossEntropyLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		crossEntropyGradient(context, gradient, output, target);
	}
	Json CrossEntropyLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { "name", "CrossEntropyLoss" });
	}
	void CrossEntropyLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "CrossEntropyLoss");
	}
} /* namespace ml */

