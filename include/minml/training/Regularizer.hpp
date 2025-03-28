/*
 * Regularizer.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_REGULARIZER_HPP_
#define MINML_TRAINING_REGULARIZER_HPP_

#include <minml/core/Tensor.hpp>

#include <vector>

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Parameter;
}

namespace ml
{
	class RegularizerL2
	{
			float m_scale = 0.0f;
		public:
			RegularizerL2(float scale = 0.0f);
			float getScale() const noexcept;
			void apply(const Context &context, float alpha, const std::vector<Tensor> &weights, std::vector<Tensor> &gradients);
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_REGULARIZER_HPP_ */
