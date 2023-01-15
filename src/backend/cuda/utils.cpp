/*
 * utils.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include "utils.hpp"
#include <minml/backend/cuda_backend.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <memory>
#include <cassert>

namespace
{
	ml::cuda::Context* get(ml::mlContext_t context)
	{
		return reinterpret_cast<ml::cuda::Context*>(context);
	}
}

namespace ml
{
	namespace cuda
	{
		Context::Context(int device_index) :
				m_device_index(device_index)
		{
		}
		Context::~Context()
		{
			if (m_cublas_handle != nullptr)
			{
				cublasStatus_t err = cublasDestroy_v2(m_cublas_handle);
				assert(err == CUBLAS_STATUS_SUCCESS);
			}
			if (m_cuda_stream != nullptr)
			{
				cudaError_t status = cudaStreamDestroy(m_cuda_stream);
				assert(status == cudaSuccess);
			}
		}
		int Context::getDeviceIndex(mlContext_t context)
		{
			if (context == nullptr)
				return -1;
			else
				return get(context)->m_device_index;
		}
		void* Context::getWorkspace(mlContext_t context)
		{
			if (context == nullptr)
				return nullptr;
			else
			{
				if (get(context)->m_workspace == nullptr)
					get(context)->m_workspace = cuda_malloc(get(context)->m_device_index, default_workspace_size);
				return get(context)->m_workspace;
			}
		}
		size_t Context::getWorkspaceSize(mlContext_t context)
		{
			if (context == nullptr)
				return 0;
			else
				return get(context)->m_workspace_size;
		}

		void Context::use(mlContext_t context)
		{
			if (context != nullptr)
			{
				cudaError_t status = cudaSetDevice(get(context)->m_device_index);
				assert(status == cudaSuccess);
			}
		}
		cudaStream_t Context::getStream(mlContext_t context)
		{
			if (context == nullptr)
				return nullptr;
			else
			{
				use(context);
				if (get(context)->m_cuda_stream == nullptr)
				{
					cudaError_t status = cudaStreamCreate(&(get(context)->m_cuda_stream));
					assert(status == cudaSuccess);
				}
				return get(context)->m_cuda_stream;
			}
		}
		cublasHandle_t Context::getHandle(mlContext_t context)
		{
			if (context == nullptr)
				return nullptr;
			else
			{
				use(context);
				if (get(context)->m_cublas_handle == nullptr)
				{
					cublasStatus_t status = cublasCreate_v2(&(get(context)->m_cublas_handle));
					assert(status == CUBLAS_STATUS_SUCCESS);
					status = cublasSetStream_v2(get(context)->m_cublas_handle, getStream(context));
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				return get(context)->m_cublas_handle;
			}
		}

	} /* namespace cuda */
} /* namespace ml */

