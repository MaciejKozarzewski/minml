/*
 * cpu_x86.hpp
 *
 *  Created on: Mar 29, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_CPU_X86_HPP_
#define BACKEND_CPU_CPU_X86_HPP_

#include <string>
#include <map>

namespace ml
{
	namespace cpu
	{
		class cpu_x86
		{
				std::string m_vendor;
				std::string m_model;
				int64_t m_memory = 0; //RAM [bytes]
				int m_cores = 0;

				struct Flag
				{
						std::string comment;
						bool is_supported = false;
				};

				std::map<std::string, Flag> m_features;
				void add_flag(const std::string &name, bool isSupported, std::string comment = "");
				void detect();
			public:
				cpu_x86();
				bool supports(const std::string &feature) const;
				const std::string& vendor() const noexcept;
				const std::string& model() const noexcept;
				int64_t memory() const noexcept;
				int64_t cores() const noexcept;
				std::string to_string() const;
				static const cpu_x86& get();
		};
	}
}

#endif /* BACKEND_CPU_CPU_X86_HPP_ */
