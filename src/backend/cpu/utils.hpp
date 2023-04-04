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

		struct ComputeConfig
		{
				mlDataType_t data_type;
				mlDataType_t compute_type;
		};

		enum class TypeSupport
		{
			NONE,
			EMULATED_CONVERSION,
			NATIVE_CONVERSION,
			NATIVE_ARITHMETIC,
			NATIVE_FMA
		};

		TypeSupport support_for_type(mlDataType_t dtype);

		bool has_hardware_fp16_conversion();

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
				WorkspaceAllocator(mlContext_t context) noexcept;
				void* get(size_t size, size_t alignment) noexcept;
		};
	}
} /* namespace ml */

#endif /* BACKEND_CPU_UTILS_HPP_ */
