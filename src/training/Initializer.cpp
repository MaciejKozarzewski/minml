/*
 * Initializer.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/Initializer.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/random.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_memory.hpp>

#include <cmath>

namespace ml
{
	void Initializer::init_weights(const Context &context, Parameter &weights, float scale, float offset)
	{
		std::vector<float> tmp(weights.shape().volume());
		for (size_t i = 0; i < tmp.size(); i++)
			tmp[i] = offset + randGaussian() * scale;
		ml::memcpy(weights.device(), weights.getParam().data(), 0, Device::cpu(), tmp.data(), 0, sizeof(float) * tmp.size());
	}
	void Initializer::init_bias(const Context &context, Parameter &bias, float scale, float offset)
	{
		std::vector<float> tmp(bias.shape().volume());
		for (size_t i = 0; i < tmp.size(); i++)
			tmp[i] = offset + randFloat() * scale;
		ml::memcpy(bias.device(), bias.getParam().data(), 0, Device::cpu(), tmp.data(), 0, sizeof(float) * tmp.size());
	}

	Json Initializer::serialize(SerializedObject &binary_data) const
	{
		return Json();
	}
	void Initializer::unserialize(const Json &json, const SerializedObject &binary_data)
	{
	}

} /* namespace ml */

