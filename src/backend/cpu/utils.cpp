/*
 * utils.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include "utils.hpp"
#include "cpu_x86.hpp"

#include <memory>
#include <stdexcept>
#include <functional>
#include <cassert>

namespace
{
	ml::cpu::Context* get(ml::mlContext_t context)
	{
		return reinterpret_cast<ml::cpu::Context*>(context);
	}
}

namespace ml
{
	namespace cpu
	{

		bool is_aligned(const void *ptr, size_t alignment) noexcept
		{
			return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
		}

		bool has_hardware_fp16_conversion()
		{
			static const bool result = cpu_x86::get().supports("f16c") or cpu_x86::get().supports("avx512-f");
			return result;
		}
		bool has_hardware_fp16_math()
		{
			static const bool result = cpu_x86::get().supports("avx512-fp16");
			return result;
		}

		Context::Context() :
				m_simd_level(getSimdSupport())
		{
		}
		SimdLevel Context::getSimdLevel(mlContext_t context)
		{
			if (context == nullptr)
				return SimdLevel::NONE;
			else
				return get(context)->m_simd_level;
		}
		void* Context::getWorkspace(mlContext_t context)
		{
			if (context == nullptr)
				return nullptr;
			else
			{
				if (get(context)->m_workspace == nullptr)
					get(context)->m_workspace = std::make_unique<uint8_t[]>(default_workspace_size);
				return get(context)->m_workspace.get();
			}
		}
		size_t Context::getWorkspaceSize(mlContext_t context)
		{
			if (context == nullptr)
				return 0;
			else
				return get(context)->m_workspace_size;
		}

	} /* namespace cpu */
} /* namespace ml */

