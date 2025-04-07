/*
 * cuda_memory.cu
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>
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
		const int elements = dstSize / sizeof(T);
		dim3 blockDim(256);
		dim3 gridDim((elements + 255) / 256);

		T v;
		std::memcpy(&v, value, sizeof(T));
		kernel_setall<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<T>(dst), elements, v);
		assert(cudaGetLastError() == cudaSuccess);

		if (stream == 0)
		{ // if launched from default stream this operation must be synchronous
			cudaError_t status = cudaDeviceSynchronize();
			assert(status == cudaSuccess);
		}
	}

	__global__ void kernel_get_cuda_arch(void *dst)
	{
		int *ptr = reinterpret_cast<int*>(dst);
#if __CUDA_ARCH__ == 100
		ptr[0] = 100;
#elif __CUDA_ARCH__ == 110
		ptr[0] = 110;
#elif __CUDA_ARCH__ == 120
		ptr[0] = 120;
#elif __CUDA_ARCH__ == 130
		ptr[0] = 130;
#elif __CUDA_ARCH__ == 200
		ptr[0] = 200;
#elif __CUDA_ARCH__ == 210
		ptr[0] = 210;
#elif __CUDA_ARCH__ == 300
		ptr[0] = 300;
#elif __CUDA_ARCH__ == 320
		ptr[0] = 320;
#elif __CUDA_ARCH__ == 350
		ptr[0] = 350;
#elif __CUDA_ARCH__ == 370
		ptr[0] = 370;
#elif __CUDA_ARCH__ == 500
		ptr[0] = 500;
#elif __CUDA_ARCH__ == 520
		ptr[0] = 520;
#elif __CUDA_ARCH__ == 530
		ptr[0] = 530;
#elif __CUDA_ARCH__ == 600
		ptr[0] = 600;
#elif __CUDA_ARCH__ == 610
		ptr[0] = 610;
#elif __CUDA_ARCH__ == 620
		ptr[0] = 620;
#elif __CUDA_ARCH__ == 700
		ptr[0] = 700;
#elif __CUDA_ARCH__ == 720
		ptr[0] = 720;
#elif __CUDA_ARCH__ == 750
		ptr[0] = 750;
#elif __CUDA_ARCH__ == 800
		ptr[0] = 800;
#elif __CUDA_ARCH__ == 860
		ptr[0] = 860;
#elif __CUDA_ARCH__ == 870
		ptr[0] = 870;
#elif __CUDA_ARCH__ == 890
		ptr[0] = 890;
#elif __CUDA_ARCH__ == 900
		ptr[0] = 900;
#else
		ptr[0] = 0;
#endif
	}

#ifndef NDEBUG
	bool is_aligned(const void *ptr, int bytes) noexcept
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % bytes) == 0;
	}
#endif
}

namespace ml
{
	namespace cuda
	{
		int get_cuda_arch(int device_index)
		{
			void *ptr = cuda_malloc(device_index, 4);
			cudaError_t status = cudaSetDevice(device_index);
			assert(status == cudaSuccess);
			kernel_get_cuda_arch<<<1, 1>>>(ptr);
			status = cudaDeviceSynchronize();
			assert(status == cudaSuccess);
			int result = 0;
			cuda_memcpy_to_host(nullptr, &result, ptr, 0, 4);
			status = cudaDeviceSynchronize();
			assert(status == cudaSuccess);
			cuda_free(ptr);

			return result;
		}
	}

	void* cuda_malloc(int device_index, size_t count)
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
	void cuda_page_lock(void *ptr, size_t count)
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
	void* cuda_create_view(void *src, size_t offset, size_t count)
	{
		if (src == nullptr)
			return nullptr;
		else
			return getPointer<uint8_t>(src) + offset;
	}
	void cuda_destroy_view(void *ptr)
	{
	}

	void cuda_memset(mlContext_t context, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count)
	{
		assert(dst != nullptr);
		void *tmp_dst = getPointer<uint8_t>(dst) + dst_offset;
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
				case 1:
					setall_launcher<uint8_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
				case 2:
					setall_launcher<uint16_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
				case 4:
					setall_launcher<uint32_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
				case 8:
					setall_launcher<uint64_t>(cuda::Context::getStream(context), tmp_dst, dst_count, src);
					break;
			}
		}
	}
	void cuda_memcpy_within_device(mlContext_t context, void *dst, size_t dst_offset, const void *src, size_t src_offset, size_t count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (context == nullptr)
		{
			cudaError_t status = cudaMemcpy(getPointer<uint8_t>(dst) + dst_offset, getPointer<uint8_t>(src) + src_offset, count,
					cudaMemcpyDeviceToDevice);
			assert(status == cudaSuccess);
		}
		else
		{
			cudaError_t status = cudaMemcpyAsync(getPointer<uint8_t>(dst) + dst_offset, getPointer<uint8_t>(src) + src_offset, count,
					cudaMemcpyDeviceToDevice, cuda::Context::getStream(context));
			assert(status == cudaSuccess);

		}
	}
	void cuda_memcpy_from_host(mlContext_t context, void *dst, size_t dst_offset, const void *src, size_t count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (context == nullptr)
		{
			cudaError_t status = cudaMemcpy(getPointer<uint8_t>(dst) + dst_offset, src, count, cudaMemcpyHostToDevice);
			assert(status == cudaSuccess);
		}
		else
		{
			cudaError_t status = cudaMemcpyAsync(getPointer<uint8_t>(dst) + dst_offset, src, count, cudaMemcpyHostToDevice,
					cuda::Context::getStream(context));
			assert(status == cudaSuccess);

		}
	}
	void cuda_memcpy_to_host(mlContext_t context, void *dst, const void *src, size_t src_offset, size_t count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (context == nullptr)
		{
			cudaError_t status = cudaMemcpy(dst, getPointer<const uint8_t>(src) + src_offset, count, cudaMemcpyDeviceToHost);
			assert(status == cudaSuccess);
		}
		else
		{
			cudaError_t status = cudaMemcpyAsync(dst, getPointer<const uint8_t>(src) + src_offset, count, cudaMemcpyDeviceToHost,
					cuda::Context::getStream(context));
			assert(status == cudaSuccess);

		}
	}

} /* namespace ml */

