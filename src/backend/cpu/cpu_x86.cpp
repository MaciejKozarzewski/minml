/*
 * cpu_x86.cpp
 *
 *  Created on: Mar 29, 2023
 *      Author: Maciej Kozarzewski
 */

#include "cpu_x86.hpp"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cassert>

#ifdef _WIN32
#  include <windows.h>
#elif MACOS
#  include <sys/param.h>
#  include <sys/sysctl.h>
#else
#  include <unistd.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#  ifdef _WIN32
#    include <Windows.h>
#    include <intrin.h>
#  elif (defined(__GNUC__) || defined(__clang__))
#    include <cpuid.h>
#  else
#    // error
#  endif
#endif

namespace
{
	template<typename T>
	inline std::string int_to_hex(T val, size_t width = sizeof(T) * 2)
	{
		std::stringstream ss;
		ss << "0x" << std::setfill('0') << std::setw(width) << std::hex << (val | 0);
		return ss.str();
	}

	class CpuID
	{
			uint32_t m_data[4] = { 0u, 0u, 0u, 0u }; // eax, ebx, ecx, edx
		public:
			class register_ref
			{
					uint32_t m_data;
				public:
					register_ref(uint32_t reg = 0u) noexcept :
							m_data(reg)
					{
					}
					operator uint32_t() const noexcept
					{
						return m_data;
					}
					bool bit(uint32_t b) const noexcept
					{
						return (m_data >> b) & 1u;
					}
					std::string toString() const;
			};
			CpuID() noexcept = default;
			CpuID(uint32_t eax, uint32_t ecx) noexcept
			{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#  if _WIN32
				__cpuidex(reinterpret_cast<int*>(m_data), eax, ecx);
#  elif (defined(__GNUC__) || defined(__clang__))
				__cpuid_count(eax, ecx, m_data[0], m_data[1], m_data[2], m_data[3]);
#  endif
#endif
			}
			register_ref eax() const noexcept
			{
				return m_data[0];
			}
			register_ref ebx() const noexcept
			{
				return m_data[1];
			}
			register_ref ecx() const noexcept
			{
				return m_data[2];
			}
			register_ref edx() const noexcept
			{
				return m_data[3];
			}
	};

	int64_t xgetbv(uint32_t index)
	{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#  if _WIN32
		return _xgetbv(index);
#  elif (defined(__GNUC__) || defined(__clang__))
		uint32_t eax, edx;
		__asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
		return (static_cast<uint64_t>(edx) << 32) | eax;
#  endif
#endif
	}

	int get_number_of_cores()
	{
#if defined(_WIN32)
		SYSTEM_INFO systeminfo;
		GetSystemInfo(&systeminfo);
		return systeminfo.dwNumberOfProcessors;
#else
		return sysconf( _SC_NPROCESSORS_ONLN);
#endif
	}

	uint64_t get_total_system_memory()
	{
#if defined(_WIN32)
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		GlobalMemoryStatusEx(&status);
		return status.ullTotalPhys;
#else
		uint64_t pages = sysconf(_SC_PHYS_PAGES);
		uint64_t page_size = sysconf(_SC_PAGE_SIZE);
		return pages * page_size;
#endif /* defined(_WIN32) */
	}
#ifndef _XCR_XFEATURE_ENABLED_MASK
#  define _XCR_XFEATURE_ENABLED_MASK  0
#endif
	bool detect_os_avx()
	{
		// Copied from: http://stackoverflow.com/a/22521619/922184
		bool avxSupported = false;

		const CpuID info(1, 0);

		const bool osUsesXSAVE_XRSTORE = info.ecx().bit(27);
		const bool cpuAVXSuport = info.ecx().bit(28);

		if (osUsesXSAVE_XRSTORE and cpuAVXSuport)
		{
			const uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
			avxSupported = (xcrFeatureMask & 0x6) == 0x6;
		}

		return avxSupported;
	}
	bool detect_os_avx512()
	{
		if (not detect_os_avx())
			return false;

		const uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
		return (xcrFeatureMask & 0xe6) == 0xe6;
	}

	std::string to_string(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
	{
		constexpr size_t size = sizeof(uint32_t) * 4;
		char tmp[size + 1];
		std::memcpy(tmp + 0, &r0, sizeof(r0));
		std::memcpy(tmp + 4, &r1, sizeof(r0));
		std::memcpy(tmp + 8, &r2, sizeof(r0));
		std::memcpy(tmp + 12, &r3, sizeof(r0));
		tmp[size] = '\0';
		return std::string(tmp);
	}
}

namespace ml
{
	namespace cpu
	{
		void cpu_x86::add_flag(const std::string &name, bool isSupported, std::string comment)
		{
			assert(m_features.find(name) == m_features.end());
			m_features.insert( { name, Flag { comment, isSupported } });
		}
		void cpu_x86::detect()
		{
			m_cores = get_number_of_cores();
			m_memory = get_total_system_memory();
			add_flag("os_avx", detect_os_avx());
			add_flag("os_avx512", detect_os_avx512());

			CpuID info;

			info = CpuID(0, 0); // EAX=0: Highest Function Parameter and Manufacturer ID
			const uint32_t max_eax = info.eax();
			m_vendor = to_string(info.ebx(), info.edx(), info.ecx(), 0);

			if (max_eax >= 0x00000001) // EAX=1: Processor Info and Feature Bits
			{
				info = CpuID(0x00000001, 0);
				add_flag("cmov", info.edx().bit(15));
				add_flag("mmx", info.edx().bit(23));
				add_flag("sse", info.edx().bit(24));
				add_flag("sse2", info.edx().bit(25));
				add_flag("htt", info.edx().bit(28), "Hyper-threading");

				add_flag("sse3", info.ecx().bit(0));
				add_flag("ssse3", info.ecx().bit(9));
				add_flag("fma3", info.ecx().bit(12));
				add_flag("sse4.1", info.ecx().bit(19));
				add_flag("sse4.2", info.ecx().bit(20));
				add_flag("popcnt", info.ecx().bit(23));
				add_flag("aes", info.ecx().bit(25));
				add_flag("xsave", info.ecx().bit(26));
				add_flag("osxsave", info.ecx().bit(27), "XSAVE enabled by OS");
				add_flag("avx", info.ecx().bit(28));
				add_flag("f16c", info.ecx().bit(29), "F16C (half-precision) FP feature");
			}

			if (max_eax >= 0x00000002) // EAX=2: Cache and TLB Descriptor information
			{
			}

			if (max_eax >= 0x00000003) // EAX=3: Processor Serial Number
			{
			}

			if (max_eax >= 0x00000004) // EAX=4 and EAX=Bh: Intel thread/core and cache topology
			{
			}

			if (max_eax >= 0x00000006) // EAX=6: Thermal and power management
			{
				info = CpuID(0x00000006, 0);
				add_flag("turbo_boost", info.eax().bit(1), "Intel Turbo Boost");
			}

			if (max_eax >= 0x00000007) // EAX=7, ECX=0: Extended Features
			{
				info = CpuID(0x00000007, 0);
				add_flag("bmi1", info.ebx().bit(3), "Bit Manipulation Instruction Set 1");
				add_flag("avx2", info.ebx().bit(5));
				add_flag("bmi2", info.ebx().bit(8), "Bit Manipulation Instruction Set 2");
				add_flag("avx512-f", info.ebx().bit(16));
				add_flag("avx512-dq", info.ebx().bit(17));
				add_flag("avx512-pf", info.ebx().bit(26));
				add_flag("avx512-er", info.ebx().bit(27));
				add_flag("avx512-cd", info.ebx().bit(28));
				add_flag("sha", info.ebx().bit(29));
				add_flag("avx512-bw", info.ebx().bit(30));
				add_flag("avx512-vl", info.ebx().bit(31));

				add_flag("prefetchwtq", info.ecx().bit(0));
				add_flag("avx512-vbmi", info.ecx().bit(1));
				add_flag("avx512-vbmi2", info.ecx().bit(6));
				add_flag("gfni", info.ecx().bit(8), "Galois Field instructions");
				add_flag("avx512-vnni", info.ecx().bit(11));
				add_flag("avx512-bitalg", info.ecx().bit(12));
				add_flag("avx512-vpopcntdq", info.ecx().bit(14));

				add_flag("avx512-4vnniw", info.edx().bit(2));
				add_flag("avx512-4fmaps", info.edx().bit(3));
				add_flag("avx512-vp2intersect", info.edx().bit(8));
				add_flag("amx-bf16", info.edx().bit(22));
				add_flag("avx512-fp16", info.edx().bit(23));
				add_flag("amx-tile", info.edx().bit(24));
				add_flag("amx-int8", info.edx().bit(25));
			}
			if (max_eax >= 0x00000007) // EAX=7, ECX=1: Extended Features
			{
				info = CpuID(0x00000007, 1);
				add_flag("avx-vnni", info.eax().bit(4));
				add_flag("avx512-bf16", info.eax().bit(5));
				add_flag("amx-fp16", info.eax().bit(21));
				add_flag("avx-ifma", info.eax().bit(23));

				add_flag("avx-vnn-int8", info.edx().bit(4));
				add_flag("avx-ne-convert", info.edx().bit(5));
			}

			// extended features
			info = CpuID(0x80000000, 0);
			const uint32_t max_extended_eax = info.eax();

			if (max_extended_eax >= 0x80000001) // EAX=0x80000001: Extended Processor Info and Feature Bits
			{
				info = CpuID(0x80000001, 0);
				add_flag("abm", info.ecx().bit(5), "Advanced bit manipulation (lzcnt and popcnt)");
				add_flag("sse4a", info.ecx().bit(6));
				add_flag("xop", info.ecx().bit(11));
				add_flag("fma4", info.ecx().bit(16));
			}

			if (max_extended_eax >= 0x80000004) // EAX=0x80000002,0x80000003,0x80000004: Processor Brand String
			{
				info = CpuID(0x80000002, 0);
				m_model = to_string(info.eax(), info.ebx(), info.ecx(), info.edx());
				info = CpuID(0x80000003, 0);
				m_model += to_string(info.eax(), info.ebx(), info.ecx(), info.edx());
				info = CpuID(0x80000004, 0);
				m_model += to_string(info.eax(), info.ebx(), info.ecx(), info.edx());
			}
		}
		cpu_x86::cpu_x86()
		{
			detect();
		}
		bool cpu_x86::supports(const std::string &feature) const
		{
			const auto iter = m_features.find(feature);
			return (iter == m_features.end()) ? false : iter->second.is_supported;
		}
		const std::string& cpu_x86::vendor() const noexcept
		{
			return m_vendor;
		}
		const std::string& cpu_x86::model() const noexcept
		{
			return m_model;
		}
		int64_t cpu_x86::memory() const noexcept
		{
			return m_memory;
		}
		int64_t cpu_x86::cores() const noexcept
		{
			return m_cores;
		}
		void cpu_x86::print() const
		{
			std::cout << "Vendor : " << vendor() << '\n';
			std::cout << "Model  : " << model() << '\n';
			std::cout << "Cores  : " << cores() << '\n';
			std::cout << "Memory : " << (memory() >> 20) << "MB\n";
			std::cout << "Features:\n";
//			size_t max_name_length = 0;
//			for (auto iter = m_features.begin(); iter != m_features.end(); iter++)
//				max_name_length = std::max(max_name_length, iter->first.length());
			for (auto iter = m_features.begin(); iter != m_features.end(); iter++)
			{
//				std::cout << std::string(max_name_length - iter->first.length(), ' ');
				std::cout << iter->first << " : ";
				std::cout << (iter->second.is_supported ? "YES" : "NO");
				if (not iter->second.comment.empty())
					std::cout << " - " + iter->second.comment;
				std::cout << '\n';
			}
		}

		const cpu_x86& cpu_x86::get()
		{
			static const cpu_x86 features;
			return features;
		}

	} /* namespace cpu */
} /* namespace ml */

