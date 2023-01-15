/*
 * conversion.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"

#include "../vectors/vectors.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cassert>

namespace
{
	template<typename T>
	__device__ T one()
	{
		return static_cast<T>(1.0f);
	}
	template<typename T>
	__device__ T zero()
	{
		return static_cast<T>(0.0f);
	}

	template<typename T>
	__global__ void kernel_unpack_input(T *output, const uint32_t *input, int first_dim)
	{
		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			const uint32_t value = input[i] >> threadIdx.x;
			output[i * 32 + threadIdx.x] = (value & 1) ? one<T>() : zero<T>();
		}
	}

	template<typename T, typename U>
	__global__ void kernel_convert(T *output, const U *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			output[i] = static_cast<T>(static_cast<float>(input[i]));
	}
}

namespace ml
{
	void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		dim3 gridDim = std::max(first_dim, 4096);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				kernel_unpack_input<<<gridDim, 32, 0, stream>>>(getPointer<half>(dst), getPointer<uint32_t>(src), first_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input<<<gridDim, 32, 0, stream>>>(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, 256);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_FLOAT16:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16>(dst), getPointer<half>(src), elements);
						break;
					case DTYPE_FLOAT32:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16>(dst), getPointer<float>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			case DTYPE_FLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<__nv_bfloat16>(src), elements);
						break;
					case DTYPE_FLOAT32:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<float>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			case DTYPE_FLOAT32:
			{
				switch (src_dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<__nv_bfloat16>(src), elements);
						break;
					case DTYPE_FLOAT16:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<half>(src), elements);
						break;
					default:
						break;
				}
				break;
			}
			default:
				break;
		}

//		if (dst_dtype == DTYPE_FLOAT16 and src_dtype == DTYPE_FLOAT32)
//			kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<float>(src), elements);
//		if (dst_dtype == DTYPE_FLOAT32 and src_dtype == DTYPE_FLOAT16)
//			kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<half>(src), elements);
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

