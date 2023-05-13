/*
 * utils.hpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_UTILS_HPP_
#define BACKEND_CPU_UTILS_HPP_

#include <minml/backend/backend_types.h>

#include <memory>
#include <string>

namespace ml
{
	namespace cpu
	{
		enum class SimdLevel
		{
			NONE,
			SSE,
			SSE2,
			SSE3,
			SSSE3,
			SSE41,
			SSE42,
			AVX,
			AVX2,
			AVX512F,
			AVX512VL_BW_DQ
		};

		std::string toString(SimdLevel sl);
		SimdLevel getSimdSupport() noexcept;

		bool has_hardware_fp16_conversion();
		bool has_hardware_bf16_conversion();

		bool has_hardware_fp16_math();
		bool has_hardware_bf16_math();

		bool is_aligned(const void *ptr, size_t alignment) noexcept;

		class Context
		{
				static constexpr size_t default_workspace_size = 8 * 1024 * 1024; // 8MB

				std::unique_ptr<uint8_t[]> m_workspace;
				size_t m_workspace_size = default_workspace_size;
				SimdLevel m_simd_level;
			public:
				Context();

				static SimdLevel getSimdLevel(mlContext_t context);
				static void* getWorkspace(mlContext_t context);
				template<typename T>
				static T* getWorkspace(mlContext_t context)
				{
					return reinterpret_cast<T*>(getWorkspace(context));
				}
				static size_t getWorkspaceSize(mlContext_t context);
		};

		class WorkspaceAllocator
		{
				void *m_ptr = nullptr;
				size_t m_max_size = 0;
				size_t m_offset = 0;
			public:
				WorkspaceAllocator() noexcept = default;
				WorkspaceAllocator(mlContext_t context) noexcept :
						m_ptr(cpu::Context::getWorkspace(context)),
						m_max_size(cpu::Context::getWorkspaceSize(context))
				{
				}
				void* get(size_t size, size_t alignment) noexcept
				{
					uint8_t *ptr = reinterpret_cast<uint8_t*>(m_ptr) + m_offset;

					const size_t remainder = reinterpret_cast<std::uintptr_t>(ptr) % alignment;
					const size_t shift = (remainder == 0) ? 0 : (alignment - remainder);

					m_offset += shift + size;
					return ptr + shift;
				}
		};
	}
} /* namespace ml */

#endif /* BACKEND_CPU_UTILS_HPP_ */
