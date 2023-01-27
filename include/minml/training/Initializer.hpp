/*
 * Initializer.hpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_INITIALIZER_HPP_
#define MINML_TRAINING_INITIALIZER_HPP_

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Parameter;
	class Context;
}

namespace ml
{

	class Initializer
	{
		public:
			void init_weights(const Context &context, Parameter &weights, float scale, float offset);
			void init_bias(const Context &context, Parameter &bias, float scale, float offset);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_INITIALIZER_HPP_ */
