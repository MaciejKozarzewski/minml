/*
 * device_context.hpp
 *
 *  Created on: Jun 7, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_CONTEXT_HPP_
#define MINML_CORE_CONTEXT_HPP_

#include <minml/core/Device.hpp>

namespace ml /* forward declarations */
{
	class Event;
}

namespace ml
{
	class Context
	{
			void *m_data = nullptr;
			Device m_device;
		public:
			Context(Device device = Device::cpu());
			Context(const Context &other) = delete;
			Context(Context &&other) noexcept;
			Context& operator=(const Context &other) = delete;
			Context& operator=(Context &&other) noexcept;
			~Context();

			Device device() const noexcept;
			bool isSynchronized() const noexcept;
			void synchronize() const;
			bool isReady() const;
			void* backend() const noexcept;
			Event createEvent() const;

			void enableTF32(bool b) noexcept;
	};

	class ContextError: public std::logic_error
	{
		public:
			ContextError(const char *function);
			ContextError(const char *function, const std::string &comment);
	};

} /* namespace ml */

#endif /* MINML_CORE_CONTEXT_HPP_ */
