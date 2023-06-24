/*
 * cpu_properties.cpp
 *
 *  Modified on: May 12, 2020
 *      by Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include "utils.hpp"
#include "cpu_x86.hpp"

#include <string>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cassert>
//#include <omp.h>

#ifdef USE_OPENBLAS
#  ifdef __linux__
#    include <cblas.h>
#  else
#    include <openblas/cblas.h>
#  endif
#endif

namespace
{
	using namespace ml;
	using namespace ml::cpu;
	SimdLevel check_supported_simd_level()
	{
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ))
//			return SimdLevel::AVX512VL_BW_DQ;
//		if (cpu_x86::get().supports("avx512-f") and cpu_x86::get().supports("os_avx512"))
//			return SimdLevel::AVX512F;
		if (cpu_x86::get().supports("avx2") and cpu_x86::get().supports("os_avx") and cpu_x86::get().supports("fma3"))
			return SimdLevel::AVX2;
		if (cpu_x86::get().supports("avx") and cpu_x86::get().supports("os_avx"))
			return SimdLevel::AVX;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE42))
//			return SimdLevel::SSE42;
		if (cpu_x86::get().supports("sse4.1"))
			return SimdLevel::SSE41;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSSE3))
//			return SimdLevel::SSSE3;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE3))
//			return SimdLevel::SSE3;
		if (cpu_x86::get().supports("sse2"))
			return SimdLevel::SSE2;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE))
//			return SimdLevel::SSE;
		return SimdLevel::NONE;
	}
	std::string get_device_info()
	{
		std::string result = cpu_x86::get().model().empty() ? cpu_x86::get().vendor() : cpu_x86::get().model();
		result += " : " + std::to_string(cpu_x86::get().cores()) + " x " + toString(check_supported_simd_level()) + " with "
				+ std::to_string(cpu_x86::get().memory() >> 20) + "MB of memory";
		return result;
	}
}

namespace ml
{
	namespace cpu
	{
		SimdLevel getSimdSupport() noexcept
		{
			static const SimdLevel supported_simd_level = check_supported_simd_level();
			return supported_simd_level;
		}

		std::string toString(SimdLevel sl)
		{
			switch (sl)
			{
				default:
				case SimdLevel::NONE:
					return "NONE";
				case SimdLevel::SSE:
					return "SSE";
				case SimdLevel::SSE2:
					return "SSE2";
				case SimdLevel::SSE3:
					return "SSE3";
				case SimdLevel::SSSE3:
					return "SSSE3";
				case SimdLevel::SSE41:
					return "SSE41";
				case SimdLevel::SSE42:
					return "SSE42";
				case SimdLevel::AVX:
					return "AVX";
				case SimdLevel::AVX2:
					return "AVX2";
				case SimdLevel::AVX512F:
					return "AVX512F";
				case SimdLevel::AVX512VL_BW_DQ:
					return "AVX512VL_BW_DQ";
			}
		}

	} /* namespace cpu */

	void cpu_set_number_of_threads(int number)
	{
//		omp_set_num_threads(number);
//#ifdef USE_OPENBLAS
//		openblas_set_num_threads(number);
//#endif
	}
	int cpu_get_number_of_cores()
	{
		return cpu_x86::get().cores();
	}
	int cpu_get_memory()
	{
		return cpu_x86::get().memory() >> 20;
	}
	int cpu_get_simd_level()
	{
		return static_cast<int>(getSimdSupport());
	}
	bool cpu_supports_type(mlDataType_t dtype)
	{
		switch (dtype)
		{
			case DTYPE_BFLOAT16:
				return false;
			case DTYPE_FLOAT16:
			{
#ifdef USE_OPENBLAS
				return false;
#else
				static const bool result = cpu_x86::get().supports("avx") and cpu_x86::get().supports("f16c");
				return result;
#endif
			}
			case DTYPE_FLOAT32:
				return true;
			default:
				return false;
		}
	}
	const char* cpu_get_device_info()
	{
		static const std::string info = get_device_info();
		return info.data();
	}
	void cpu_print_device_features()
	{
		cpu_x86::get().print();
		std::cout << "Detected SIMD level : " << cpu_get_simd_level() << '\n';

	}

} /* namespace ml */
