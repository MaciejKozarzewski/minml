/*
 * Parameter.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_PARAMETER_HPP_
#define MINML_LAYERS_PARAMETER_HPP_

#include <minml/core/Tensor.hpp>

#include <minml/training/Optimizer.hpp>
#include <minml/training/Regularizer.hpp>

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Shape;
	class Device;
	enum class DataType;
}

namespace ml
{
	class Parameter
	{
		private:
			Tensor m_param;
			Tensor m_gradient;

			// optimizer data
			Optimizer m_optimizer;
			Regularizer m_regularizer;

			int m_accumulated_updates = 0;
			bool m_is_trainable = true;
		public:
			Parameter(const Json &json, const SerializedObject &binary_data);
			Parameter(const Shape &shape, DataType dtype, Device device, bool trainable = true);

			void setTrainable(bool t);
			bool isTrainable() const noexcept;

			Optimizer& getOptimizer();
			Regularizer& getRegularizer();

			Shape shape() const noexcept;
			DataType dtype() const noexcept;
			Device device() const noexcept;
			double getInvBatch() const noexcept;
			int getBatch() const noexcept;

			const Tensor& getParam() const;
			Tensor& getParam();
			Tensor& getGradient();

			void moveTo(Device newDevice);
			void convertTo(const Context &context, DataType newType);
			void init(const Context &context);
			void learn(const Context &context);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_PARAMETER_HPP_ */
