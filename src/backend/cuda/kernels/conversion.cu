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
#include "../helpers/indexers.cuh"

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
	__global__ void kernel_unpack_input(T *output, const uint32_t *input, int first_dim, int last_dim)
	{
		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			const uint32_t value = input[i] >> threadIdx.x;
			output[i * last_dim + threadIdx.x] = (value & 1) ? one<T>() : zero<T>();
		}
	}

	template<typename T, typename U>
	__global__ void kernel_convert(T *output, const U *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			output[i] = static_cast<T>(static_cast<float>(input[i]));
	}

	template<typename T, int Rows, int Columns>
	__global__ void kernel_transpose_021(T *output, const T *input, int dim0, int dim1, int dim2)
	{
		__shared__ T workspace[Rows * Columns];
		constexpr int Padding = 4 / sizeof(T);

		Indexer<3> src_indexer(dim0, dim1, dim2);
		for (int row = threadIdx.y; row < Rows; row += blockDim.y)
			for (int col = threadIdx.x; col < Columns; col += blockDim.x)
			{
				const int d0 = blockIdx.x;
				const int d1 = blockIdx.y * Rows + row;
				const int d2 = blockIdx.z * Columns + col;
				if (d1 < dim1 && d2 < dim2)
				{
					const int idx = row * Columns + (row * Padding + col) % Columns;
					workspace[idx] = input[src_indexer.at(d0, d1, d2)];
				}
			}
		__syncthreads();

		Indexer<3> dst_indexer(dim0, dim2, dim1);
		for (int col = threadIdx.y; col < Columns; col += blockDim.y)
			for (int row = threadIdx.x; row < Rows; row += blockDim.x)
			{
				const int d0 = blockIdx.x;
				const int d1 = blockIdx.y * Rows + row;
				const int d2 = blockIdx.z * Columns + col;
				if (d1 < dim1 && d2 < dim2)
				{
					const int idx = row * Columns + (row * Padding + col) % Columns;
					output[dst_indexer.at(d0, d2, d1)] = workspace[idx];
				}
			}
	}
}

namespace ml
{
	void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		dim3 gridDim = std::max(first_dim, 4096);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
				kernel_unpack_input<<<gridDim, last_dim, 0, stream>>>(getPointer<__nv_bfloat16 >(dst), getPointer<uint32_t>(src), first_dim,
						last_dim);
				break;
			case DTYPE_FLOAT16:
				kernel_unpack_input<<<gridDim, last_dim, 0, stream>>>(getPointer<half>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input<<<gridDim, last_dim, 0, stream>>>(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, 256);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (dst_dtype == src_dtype && dst != src)
		{ // same type, different locations, can just copy memory
			cudaError_t status = cudaMemcpy(dst, src, elements * size_of(dst_dtype), cudaMemcpyDeviceToDevice);
			assert(status == cudaSuccess);
			return;
		}

		switch (dst_dtype)
		{
			case DTYPE_BFLOAT16:
			{
				switch (src_dtype)
				{
					case DTYPE_FLOAT16:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(dst), getPointer<half>(src), elements);
						break;
					case DTYPE_FLOAT32:
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(dst), getPointer<float>(src), elements);
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
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<__nv_bfloat16 >(src), elements);
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
						kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<__nv_bfloat16 >(src), elements);
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
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		assert(input != output);

		dim3 blockDim(64, 8);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_BFLOAT16:
			case DTYPE_FLOAT16:
			{
				dim3 gridDim(shape.dim[0], (shape.dim[1] + 128 - 1) / 128, (shape.dim[2] + 64 - 1) / 64);
				kernel_transpose_021<uint16_t, 128, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint16_t>(output), getPointer<uint16_t>(input),
						shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			}
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
			{
				dim3 gridDim(shape.dim[0], (shape.dim[1] + 64 - 1) / 64, (shape.dim[2] + 64 - 1) / 64);
				kernel_transpose_021<uint32_t, 64, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint32_t>(output), getPointer<uint32_t>(input),
						shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

