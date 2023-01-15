/*
 * CrossEntropyLoss.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_CROSSENTROPYLOSS_HPP_
#define MINML_TRAINING_CROSSENTROPYLOSS_HPP_

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Tensor;
}
namespace ml
{

	class CrossEntropyLoss
	{
		public:
			float getLoss(const Context &context,  const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_CROSSENTROPYLOSS_HPP_ */
