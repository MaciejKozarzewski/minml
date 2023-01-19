/*
 * Parameter.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Parameter.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{

	Parameter::Parameter(const Json &json, const SerializedObject &binary_data) :
			m_accumulated_updates(json["accumulated updates"]),
			m_is_trainable(json["is trainable"])
	{
		assert(json.hasKey("param") && json["param"].isNull() == false);
		m_param.unserialize(json["param"], binary_data);
		if (json.hasKey("update"))
			m_gradient.unserialize(json["update"], binary_data);
		if (json.hasKey("optimizer"))
			m_optimizer.unserialize(json["optimizer"], binary_data);
		if (json.hasKey("regularizer"))
			m_regularizer.unserialize(json["regularizer"], binary_data);
	}
	Parameter::Parameter(const Shape &shape, DataType dtype, Device device, bool trainable) :
			m_param(shape, dtype, device),
			m_is_trainable(trainable)
	{
	}

	void Parameter::setTrainable(bool t)
	{
		m_is_trainable = t;
	}
	bool Parameter::isTrainable() const noexcept
	{
		return m_is_trainable;
	}

	Optimizer& Parameter::getOptimizer()
	{
		return m_optimizer;
	}
	Regularizer& Parameter::getRegularizer()
	{
		return m_regularizer;
	}

	Shape Parameter::shape() const noexcept
	{
		return getParam().shape();
	}
	DataType Parameter::dtype() const noexcept
	{
		return getParam().dtype();
	}
	Device Parameter::device() const noexcept
	{
		return getParam().device();
	}
	double Parameter::getInvBatch() const noexcept
	{
		if (m_accumulated_updates == 0)
			return 0.0;
		else
			return 1.0 / m_accumulated_updates;
	}
	int Parameter::getBatch() const noexcept
	{
		return m_accumulated_updates;
	}

	const Tensor& Parameter::getParam() const
	{
		return m_param;
	}
	Tensor& Parameter::getParam()
	{
		return m_param;
	}
	Tensor& Parameter::getGradient()
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "parameter is set as non-trainable");
		if (m_gradient.isEmpty() and shape().rank() != 0)
			m_gradient = Tensor(shape(), "float32", device());

		return m_gradient;
	}

	void Parameter::moveTo(Device newDevice)
	{
		m_param.moveTo(newDevice);
		if (isTrainable())
		{
			m_gradient.moveTo(newDevice);
			m_optimizer.moveTo(newDevice);
		}
	}
	void Parameter::convertTo(const Context &context, DataType newType)
	{
		m_param.convertTo(context, newType);
	}
	void Parameter::learn(const Context &context)
	{
		if (isTrainable() and shape().rank() != 0)
		{
			getRegularizer().apply(context, *this);
			getOptimizer().apply(context, *this);
		}
	}

	Json Parameter::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["is trainable"] = m_is_trainable;
		result["accumulated updates"] = m_accumulated_updates;
		result["param"] = m_param.serialize(binary_data);
		if (isTrainable())
		{
			result["optimizer"] = m_optimizer.serialize(binary_data);
			result["regularizer"] = m_regularizer.serialize(binary_data);
		}
		return result;
	}
	void Parameter::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_param.unserialize(json["param"], binary_data);
		m_accumulated_updates = json["accumulated updates"];
		m_is_trainable = json["is trainable"];
		if (isTrainable())
		{
			m_optimizer.unserialize(json["optimizer"], binary_data);
			m_regularizer.unserialize(json["regularizer"], binary_data);
		}
	}

} /* namespace ml */

