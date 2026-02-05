/*
 * Parameter.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Parameter.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>

namespace ml
{

	Parameter::Parameter(const Shape &shape, DataType dtype, Device device) :
			m_param(shape, dtype, device)
	{
	}
	Parameter::Parameter(const Json &json, const SerializedObject &binary_data)
	{
		if (json.hasKey("param"))
			m_param.unserialize(json["param"], binary_data);
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
		if (m_gradient.isEmpty())
			m_gradient = Tensor(shape(), dtype(), device());
		return m_gradient;
	}

	void Parameter::moveTo(Device newDevice)
	{
		m_param.moveTo(newDevice);
		m_gradient.moveTo(newDevice);
	}
	void Parameter::convertTo(const Context &context, DataType newType)
	{
		m_param.convertTo(context, newType);
		m_gradient.convertTo(context, newType);
	}

	Json Parameter::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["param"] = m_param.serialize(binary_data);
		return result;
	}
	void Parameter::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_param.unserialize(json["param"], binary_data);
	}

} /* namespace ml */

