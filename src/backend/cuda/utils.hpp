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
#ifdef USE_CUDNN
#  include "cudnn_ops_infer.h"
#  include <cublasLt.h>
#endif

#include <memory>
#include <algorithm>
#include <string>

namespace ml
{
	namespace cuda
	{
		class Context
		{
				static constexpr size_t default_workspace_size = 32 * 1024 * 1024; // 32MB

				void *m_workspace = nullptr;
				size_t m_workspace_size = default_workspace_size;
				cudaStream_t m_cuda_stream = nullptr;
				cublasHandle_t m_cublas_handle = nullptr;
#ifdef USE_CUDNN
				cudnnHandle_t m_cudnn_handle = nullptr;
				cublasLtHandle_t m_cublas_lt_handle = nullptr;
#endif
				int m_device_index = 0;
				bool m_allows_tf32 = false;
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
				static void setWorkspaceSize(mlContext_t context, size_t bytes);
				static void use(mlContext_t context);
				static cudaStream_t getStream(mlContext_t context);
				static cublasHandle_t getHandle(mlContext_t context);
				static void enableTF32(mlContext_t context, bool b);
				static bool allowsTF32(mlContext_t context);
#ifdef USE_CUDNN
				static cudnnHandle_t getCudnnHandle(mlContext_t context);
				static cublasLtHandle_t getCublasLtHandle(mlContext_t context);
#endif
		};

		template<unsigned int maxBlocks>
		unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
		{
			return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
		}

		int get_compute_capability(int device_index);

		bool has_fp16_math(mlContext_t context);
		bool has_tensor_cores(mlContext_t context);
		bool allows_tf32(mlContext_t context);

		int get_cuda_arch(int device_index);

	}
} /* namespace ml */

#endif /* BACKEND_CUDA_UTILS_HPP_ */
