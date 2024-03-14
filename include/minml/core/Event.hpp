/*
 * Event.hpp
 *
 *  Created on: Nov 20, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_EVENT_HPP_
#define MINML_CORE_EVENT_HPP_

#include <minml/core/Device.hpp>

namespace ml /* forward declarations */
{
	class Context;
}

namespace ml
{
	class Event
	{
			void *m_data = nullptr;
			Device m_device = Device::cpu();
		public:
			Event() noexcept = default;
			Event(const Context &context);
			Event(const Event &other) = delete;
			Event(Event &&other) noexcept;
			Event& operator=(const Event &other) = delete;
			Event& operator=(Event &&other) noexcept;
			~Event();

			Device device() const noexcept;
			void synchronize() const;
			bool isReady() const;
			void* backend() const noexcept;

			static double getElapsedTime(const Event &start, const Event &end);
	};

} /* namespace ml */

#endif /* MINML_CORE_EVENT_HPP_ */
