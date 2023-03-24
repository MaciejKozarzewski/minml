/*
 * Regularizer.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_REGULARIZER_HPP_
#define MINML_TRAINING_REGULARIZER_HPP_

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Parameter;
}

namespace ml
{

	class Regularizer
	{
			float m_coefficient = 0.0f;
			float m_offset = 0.0f;
		public:
			Regularizer() = default;
			Regularizer(float coefficient, float offset = 0.0f);

			float getCoefficient() const noexcept;
			float getOffset() const noexcept;

			void apply(const Context &context, Parameter &param);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_REGULARIZER_HPP_ */
