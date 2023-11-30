/*
 * Event.cpp
 *
 *  Created on: Nov 20, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Event.hpp>
#include <minml/core/Context.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>
#include <minml/backend/opencl_backend.h>

#include <algorithm>

namespace ml
{
	Event::Event(const Context &context) :
			m_data(nullptr),
			m_device(context.device())
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				m_data = cpu_create_event(context.backend());
				break;
			case DeviceType::CUDA:
				m_data = cuda_create_event(context.backend());
				break;
			case DeviceType::OPENCL:
				m_data = opencl_create_event(context.backend());
				break;
		}
	}
	Event::Event(Event &&other) noexcept :
			m_data(other.m_data),
			m_device(other.m_device)
	{
		other.m_data = nullptr;
	}
	Event& Event::operator=(Event &&other) noexcept
	{
		std::swap(this->m_data, other.m_data);
		std::swap(this->m_device, other.m_device);
		return *this;
	}
	Event::~Event()
	{
		switch (device().type())
		{
			case DeviceType::CPU:
				cpu_destroy_event(backend());
				break;
			case DeviceType::CUDA:
				cuda_destroy_event(backend());
				break;
			case DeviceType::OPENCL:
				opencl_destroy_event(backend());
				break;
		}
	}
	Device Event::device() const noexcept
	{
		return m_device;
	}
	void Event::synchronize() const
	{
		switch (m_device.type())
		{
			case DeviceType::CPU:
				cpu_wait_for_event(backend());
				break;
			case DeviceType::CUDA:
				cuda_wait_for_event(backend());
				break;
			case DeviceType::OPENCL:
				opencl_wait_for_event(backend());
				break;
		}
	}
	bool Event::isReady() const
	{
		switch (m_device.type())
		{
			default:
			case DeviceType::CPU:
				return cpu_is_event_ready(backend());
			case DeviceType::CUDA:
				return cuda_is_event_ready(backend());
			case DeviceType::OPENCL:
				return opencl_is_event_ready(backend());
		}
	}
	void* Event::backend() const noexcept
	{
		return m_data;
	}

	double Event::getElapsedTime(const Event &start, const Event &end)
	{
		if (start.device() != end.device())
			throw std::runtime_error("Event::getElapsedTime() events come from different devices");
		switch (start.device().type())
		{
			default:
			case DeviceType::CPU:
				return cpu_get_time_between_events(start.backend(), end.backend());
			case DeviceType::CUDA:
				return cuda_get_time_between_events(start.backend(), end.backend());
			case DeviceType::OPENCL:
				return opencl_get_time_between_events(start.backend(), end.backend());
		}
	}

} /* namespace ml */

