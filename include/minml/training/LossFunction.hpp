/*
 * LossFunction.hpp
 *
 *  Created on: May 23, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_LOSSFUNCTION_HPP_
#define MINML_TRAINING_LOSSFUNCTION_HPP_

#include <memory>

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Tensor;
}
namespace ml
{

	class LossFunction
	{
		public:
			virtual ~LossFunction() = default;
			virtual float getLoss(const Context &context, const Tensor &output, const Tensor &target) const = 0;
			virtual void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const = 0;
			virtual Json serialize(SerializedObject &binary_data) const = 0;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data) = 0;
			virtual std::unique_ptr<LossFunction> clone() const = 0;
	};

	class CrossEntropyLoss: public LossFunction
	{
			float m_weight;
		public:
			CrossEntropyLoss(float weight = 1.0f);
			float getLoss(const Context &context, const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
			std::unique_ptr<LossFunction> clone() const;
	};

	class MeanSquaredLoss: public LossFunction
	{
			float m_weight;
		public:
			MeanSquaredLoss(float weight = 1.0f);
			float getLoss(const Context &context, const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
			std::unique_ptr<LossFunction> clone() const;
	};

	class ValueHeadLoss: public LossFunction
	{
			float m_weight;
		public:
			ValueHeadLoss(float weight = 1.0f);
			float getLoss(const Context &context, const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
			std::unique_ptr<LossFunction> clone() const;
	};

} /* namespace ml */

#endif /* MINML_TRAINING_LOSSFUNCTION_HPP_ */
