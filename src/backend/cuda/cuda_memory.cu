/*
 * cuda_memory.cu
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include "utils.hpp"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstring>
#include <iostream>
#include <cassert>

namespace
{
	using namespace ml::cuda;

	template<typename T>
	__global__ void kernel_setall(T *ptr, int length, T value)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			ptr[i] = value;
	}
	template<typename T>
	void setall_launcher(cudaStream_t stream, void *dst, int dstSize, const void *value)
	{
		const int length = dstSize / sizeof(T);
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(length, blockDim.x);

		T v;
		std::memcpy(&v, value, sizeof(T));
		kernel_setall<<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<T*>(dst), length, v);
		assert(cudaGetLastError() == cudaSuccess);

		if (stream == 0)
		{ // if launched from default stream this operation must be synchronous
			cudaError_t status = cudaDeviceSynchronize();
			assert(status == cudaSuccess);
		}
	}

	bool is_aligned(const void *ptr, int bytes) noexcept
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % bytes) == 0;
	}
}

namespace ml
{
	void* cuda_malloc(int device_index, int count)
	{
		void *result = nullptr;
		if (count > 0)
		{
			cudaError_t status = cudaSetDevice(device_index);
			assert(status == cudaSuccess);
			status = cudaMalloc(reinterpret_cast<void**>(&result), count);
			assert(status == cudaSuccess);
		}
		return result;
	}
	void cuda_page_lock(void *ptr, int count)
	{
		if (ptr != nullptr)
		{
			cudaError_t status = cudaHostRegister(ptr, count, 0);
			assert(status == cudaSuccess);
		}
	}
	void cuda_page_unlock(void *ptr)
	{
		if (ptr != nullptr)
		{
			cudaError_t status = cudaHostUnregister(ptr);
			assert(status == cudaSuccess);
		}
	}
	void cuda_free(void *ptr)
	{
		if (ptr != nullptr)
		{
			cudaError_t status = cudaFree(ptr);
			assert(status == cudaSuccess);
		}
	}
	void* cuda_view(void *src, int offset, int count)
	{
		if (src == nullptr)
			return nullptr;
		else
			return reinterpret_cast<uint8_t*>(src) + offset;
	}

	void cuda_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		void *tmp_dst = reinterpret_cast<uint8_t*>(dst) + dst_offset;
		if (src == nullptr)
		{
			if (context == nullptr)
			{
				cuda::Context::use(context);
				cudaError_t status = cudaMemset(tmp_dst, 0, dst_count);
				assert(status == cudaSuccess);
				status = cudaDeviceSynchronize(); // if launched from default stream this operation must be synchronous
				assert(status == cudaSuccess);
			}
			else
			{
				cudaError_t status = cudaMemsetAsync(tmp_dst, 0, dst_count, cuda::Context::getStream(context));
				assert(status == cudaSuccess);
			}
		}
		else
		{
			assert(dst_count % src_count == 0);
			assert(is_aligned(tmp_dst, src_count));

			switch (src_count)
			{
				case 2:
					setall_launcher<uint16_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
				case 4:
					setall_launcher<uint32_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
			}
		}
	}
	void cuda_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (context == nullptr)
		{
			cudaError_t status = cudaMemcpy(reinterpret_cast<uint8_t*>(dst) + dst_offset, src, count, cudaMemcpyHostToDevice);
			assert(status == cudaSuccess);
		}
		else
		{
			cudaError_t status = cudaMemcpyAsync(reinterpret_cast<uint8_t*>(dst) + dst_offset, src, count, cudaMemcpyHostToDevice,
					cuda::Context::getStream(context));
			assert(status == cudaSuccess);

		}
	}
	void cuda_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (context == nullptr)
		{
			cudaError_t status = cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(src) + src_offset, count, cudaMemcpyDeviceToHost);
			assert(status == cudaSuccess);
		}
		else
		{
			cudaError_t status = cudaMemcpyAsync(dst, reinterpret_cast<const uint8_t*>(src) + src_offset, count, cudaMemcpyDeviceToHost,
					cuda::Context::getStream(context));
			assert(status == cudaSuccess);

		}
	}
} /* namespace ml */

