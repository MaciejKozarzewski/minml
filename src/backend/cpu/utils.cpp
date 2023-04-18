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
		Context::Context() :
				m_simd_level(getSimdSupport())
		{
		}

		bool has_hardware_fp16_conversion()
		{
			static const bool result = cpu_x86::get().supports("f16c") or cpu_x86::get().supports("avx512-f");
			return result;
		}
		bool has_hardware_bf16_conversion()
		{
			static const bool result = cpu_x86::get().supports("avx512-f") and cpu_x86::get().supports("avx512-bf16");
			return result;
		}

		bool has_hardware_fp16_math()
		{
			static const bool result = cpu_x86::get().supports("avx512-fp16");
			return result;
		}
		bool has_hardware_bf16_math()
		{
			static const bool result = cpu_x86::get().supports("avx512-f") and cpu_x86::get().supports("avx512-bf16");
			return result;
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

		WorkspaceAllocator::WorkspaceAllocator(mlContext_t context) noexcept :
				m_ptr(cpu::Context::getWorkspace(context)),
				m_max_size(cpu::Context::getWorkspaceSize(context))
		{
			assert(context != nullptr);
		}
		void* WorkspaceAllocator::get(size_t size, size_t alignment) noexcept
		{
			assert(m_ptr != nullptr);
			const size_t shift = reinterpret_cast<std::uintptr_t>(m_ptr) % alignment;
			assert(m_offset + shift + size <= m_max_size);

			void *result = reinterpret_cast<uint8_t*>(m_ptr) + m_offset + shift;
			m_offset += shift + size;
			return result;
		}

	} /* namespace cpu */
} /* namespace ml */

