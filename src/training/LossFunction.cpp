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
	float CrossEntropyLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask) const
	{
		return crossEntropyLoss(context, output, target, mask);
	}
	void CrossEntropyLoss::getGradient(const Context &context, float scale, Tensor &gradient, const Tensor &output, const Tensor &target, const Tensor &mask) const
	{
		crossEntropyGradient(context, scale, output, target, mask, 0.0f, gradient);
	}
	Json CrossEntropyLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", "CrossEntropyLoss" } });
	}
	void CrossEntropyLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "CrossEntropyLoss");
	}
	std::unique_ptr<LossFunction> CrossEntropyLoss::clone() const
	{
		return std::make_unique<CrossEntropyLoss>();
	}

	float MeanSquaredLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask) const
	{
		return meanSquaredLoss(context, output, target, mask);
	}
	void MeanSquaredLoss::getGradient(const Context &context, float scale, Tensor &gradient, const Tensor &output, const Tensor &target, const Tensor &mask) const
	{
		meanSquaredGradient(context, scale, output, target, mask, 0.0f, gradient);
	}
	Json MeanSquaredLoss::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", "MeanSquaredLoss" } });
	}
	void MeanSquaredLoss::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		assert(json["name"].getString() == "MeanSquaredLoss");
	}
	std::unique_ptr<LossFunction> MeanSquaredLoss::clone() const
	{
		return std::make_unique<MeanSquaredLoss>();
	}


} /* namespace ml */

