/*
 * utils.hpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_UTILS_HPP_
#define BACKEND_CUDA_UTILS_HPP_

#include <minml/backend/backend_types.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <memory>
#include <algorithm>
#include <string>

namespace ml
{
	namespace cuda
	{
		class Context
		{
				static constexpr size_t default_workspace_size = 8 * 1024 * 1024; // 8MB

				void *m_workspace = nullptr;
				size_t m_workspace_size = default_workspace_size;
				cudaStream_t m_cuda_stream = nullptr;
				cublasHandle_t m_cublas_handle = nullptr;
				int m_device_index = 0;
			public:
				Context(int device_index);
				~Context();
				static int getDeviceIndex(mlContext_t context);
				static void* getWorkspace(mlContext_t context);
				template<typename T>
				static T* getWorkspace(mlContext_t context)
				{
					return reinterpret_cast<T*>(getWorkspace(context));
				}
				static size_t getWorkspaceSize(mlContext_t context);
				static void use(mlContext_t context);
				static cudaStream_t getStream(mlContext_t context);
				static cublasHandle_t getHandle(mlContext_t context);
		};

		template<unsigned int maxBlocks>
		unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
		{
			return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
		}

		int get_compute_capability(int device_index);

		bool has_fp16_math(mlContext_t context);
		bool has_bf16_math(mlContext_t context);

	}
} /* namespace ml */

#endif /* BACKEND_CUDA_UTILS_HPP_ */
