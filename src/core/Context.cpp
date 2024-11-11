/*
 * Context.cpp
 *
 *  Created on: Jun 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Event.hpp>
#include <minml/core/ml_exceptions.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>
#include <minml/backend/opencl_backend.h>

#include <algorithm>

namespace ml
{
	Context::Context(Device d) :
			m_device(d)
	{
		switch (d.type())
		{
			case DeviceType::CPU:
				m_data = cpu_create_context();
				break;
			case DeviceType::CUDA:
				m_data = cuda_create_context(d.index());
				break;
			case DeviceType::OPENCL:
				m_data = opencl_create_context(d.index());
				break;
		}
		if (m_data == nullptr)
			throw ContextError(METHOD_NAME, "failed to create context on " + d);
	}
	Context::Context(Context &&other) noexcept :
			m_data(other.m_data),
			m_device(other.m_device)
	{
		other.m_data = nullptr;
	}
	Context& Context::operator=(Context &&other) noexcept
	{
		std::swap(this->m_data, other.m_data);
		std::swap(this->m_device, other.m_device);
		return *this;
	}
	Context::~Context()
	{
		switch (device().type())
		{
			case DeviceType::CPU:
				cpu_destroy_context(backend());
				break;
			case DeviceType::CUDA:
				cuda_destroy_context(backend());
				break;
			case DeviceType::OPENCL:
				opencl_destroy_context(backend());
				break;
		}
	}
	Device Context::device() const noexcept
	{
		return m_device;
	}
	bool Context::isSynchronized() const noexcept
	{
		switch (m_device.type())
		{
			default:
			case DeviceType::CPU:
				return true;
			case DeviceType::CUDA:
				return false;
			case DeviceType::OPENCL:
				return false;
		}
	}
	void Context::synchronize() const
	{
		switch (m_device.type())
		{
			case DeviceType::CPU:
				cpu_synchronize_with_context(backend());
				break;
			case DeviceType::CUDA:
				cuda_synchronize_with_context(backend());
				break;
			case DeviceType::OPENCL:
				opencl_synchronize_with_context(backend());
				break;
		}
	}
	bool Context::isReady() const
	{
		switch (m_device.type())
		{
			case DeviceType::CPU:
				return cpu_is_context_ready(backend());
			case DeviceType::CUDA:
				return cuda_is_context_ready(backend());
			case DeviceType::OPENCL:
				return opencl_is_context_ready(backend());
			default:
				return false;
		}
	}
	void* Context::backend() const noexcept
	{
		return m_data;
	}
	Event Context::createEvent() const
	{
		return Event(*this);
	}

	void Context::enableTF32(bool b) noexcept
	{
		if (device().isCUDA())
			cuda_enable_tf32(backend(), b);
	}

	ContextError::ContextError(const char *function) :
			std::logic_error(function)
	{
	}
	ContextError::ContextError(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

} /* namespace ml */

