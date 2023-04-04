/*
 * cpu_properties.cpp
 *
 *  Created on : Dec 4, 2014
 *      Author: Alexander J. Yee (https://github.com/Mysticial/FeatureDetector)
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
#include <omp.h>

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
	ml::cpu::SimdLevel check_supported_simd_level()
	{
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ))
//			return SimdLevel::AVX512VL_BW_DQ;
		if (cpu_x86::get().supports("avx512-f") and cpu_x86::get().supports("os_avx512"))
			return ml::cpu::SimdLevel::AVX512F;
		if (cpu_x86::get().supports("avx2") and cpu_x86::get().supports("os_avx") and cpu_x86::get().supports("fma3"))
			return ml::cpu::SimdLevel::AVX2;
		if (cpu_x86::get().supports("avx") and cpu_x86::get().supports("os_avx"))
			return ml::cpu::SimdLevel::AVX;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE42))
//			return SimdLevel::SSE42;
		if (cpu_x86::get().supports("sse4.1"))
			return ml::cpu::SimdLevel::SSE41;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSSE3))
//			return SimdLevel::SSSE3;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE3))
//			return SimdLevel::SSE3;
		if (cpu_x86::get().supports("sse2"))
			return ml::cpu::SimdLevel::SSE2;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE))
//			return SimdLevel::SSE;
		return ml::cpu::SimdLevel::NONE;
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

		TypeSupport support_for_type(mlDataType_t dtype)
		{
			static const bool fma3 = cpu_x86::get().supports("fma3");
			static const bool f16c = cpu_x86::get().supports("f16c");
			static const bool avx512f = cpu_x86::get().supports("avx512-f");
			static const bool avx512_bf16 = cpu_x86::get().supports("avx512-bf16");
			static const bool avx512_fp16 = cpu_x86::get().supports("avx512-fp16");

			switch (dtype)
			{
				default:
				case DTYPE_UNKNOWN:
					return TypeSupport::NONE;
				case DTYPE_BFLOAT16:
					return (avx512f and avx512_bf16) ? TypeSupport::NATIVE_FMA : TypeSupport::EMULATED_CONVERSION;
				case DTYPE_FLOAT16:
				{
					if (avx512_fp16)
						return TypeSupport::NATIVE_FMA;
					else
						return (f16c or avx512f) ? TypeSupport::NATIVE_CONVERSION : TypeSupport::EMULATED_CONVERSION;
				}
				case DTYPE_FLOAT32:
					return fma3 ? TypeSupport::NATIVE_FMA : TypeSupport::NATIVE_ARITHMETIC;
				case DTYPE_INT32:
					return TypeSupport::NATIVE_ARITHMETIC;
			}
		}

		bool has_hardware_fp16_conversion()
		{
			static const bool result = cpu_x86::get().supports("f16c");
			return result;
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
		omp_set_num_threads(number);
#ifdef USE_OPENBLAS
		openblas_set_num_threads(number);
#endif
	}
	int cpu_get_number_of_cores()
	{
		return cpu_x86::get().cores();
	}
	int cpu_get_memory()
	{
		return cpu_x86::get().memory() >> 20;
	}
	bool cpu_supports_type(mlDataType_t dtype)
	{
		switch (dtype)
		{
			case DTYPE_BFLOAT16:
				return false;
			case DTYPE_FLOAT16:
				return false;
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

} /* namespace ml */
