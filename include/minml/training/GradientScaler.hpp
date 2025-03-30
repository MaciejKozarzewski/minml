/*
 * GradientScaler.hpp
 *
 *  Created on: Mar 23, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_GRADIENTSCALER_HPP_
#define MINML_TRAINING_GRADIENTSCALER_HPP_

#include <vector>

class Json;
class SerializedObject;
namespace ml
{
	class Context;
	class Tensor;
}

namespace ml
{
	class GradientScaler
	{
			float m_scale = 1.0f;
			int m_steps_since_scale_change = 0;
			int m_scale_change_interval = 1000;
		public:
			GradientScaler(float initialScale = 1.0f, int scaleChangeInterval = 1000);
			float getScale() const noexcept;
			float getInvScale(const Context &context, std::vector<Tensor> &params);
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_GRADIENTSCALER_HPP_ */
