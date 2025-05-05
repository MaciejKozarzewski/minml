/*
 * softmax.cu
 *
 *  Created on: Nov 3, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;

	template<typename T>
	__global__ void kernel_softmax_3_channels(T *output, const T *input, int first_dim)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < first_dim)
		{
			float x0 = static_cast<float>(input[idx * 3 + 0]);
			float x1 = static_cast<float>(input[idx * 3 + 1]);
			float x2 = static_cast<float>(input[idx * 3 + 2]);

			const float max_value = max(x0, max(x1, x2));
			x0 = exp(x0 - max_value);
			x1 = exp(x1 - max_value);
			x2 = exp(x2 - max_value);

			const float inv_sum = 1.0f / (x0 + x1 + x2);

			output[idx * 3 + 0] = static_cast<T>(x0 * inv_sum);
			output[idx * 3 + 1] = static_cast<T>(x1 * inv_sum);
			output[idx * 3 + 2] = static_cast<T>(x2 * inv_sum);
		}
	}

	template<typename T>
	__global__ void kernel_softmax_generic(T *output, const T *input, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<128> tile = cg::tiled_partition<128>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float max_value = -1e+32f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = static_cast<float>(input[i * last_dim + j]);
				max_value = max(max_value, workspace[j]);
			}
			const float shift = cg::reduce(tile, max_value, cg::greater<float>());

			float partial_sum = 0.0f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = exp(workspace[j] - shift);
				partial_sum += workspace[j];
			}
			const float inv_sum = 1.0f / cg::reduce(tile, partial_sum, cg::plus<float>());
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
				output[i * last_dim + j] = static_cast<T>(workspace[j] * inv_sum);
		}
	}
}

namespace ml
{
	void cuda_softmax_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		assert(input != nullptr);
		assert(output != nullptr);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		assert(shape.rank == 2);
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape);

		if (last_dim == 3)
		{
			dim3 blockDim(256);
			dim3 gridDim((first_dim + 255) / 256);
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_softmax_3_channels<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output), ml::getPointer<half>(input), first_dim);
					break;
				case DTYPE_FLOAT32:
					kernel_softmax_3_channels<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output), ml::getPointer<float>(input),
							first_dim);
					break;
				case DTYPE_FLOAT64:
					kernel_softmax_3_channels<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output), ml::getPointer<double>(input),
							first_dim);
					break;
			}
		}
		else
		{
			dim3 blockDim(128);
			dim3 gridDim(std::min(1024, first_dim));
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_softmax_generic<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output), ml::getPointer<half>(input), first_dim,
							last_dim);
					break;
				case DTYPE_FLOAT32:
					kernel_softmax_generic<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output), ml::getPointer<float>(input), first_dim,
							last_dim);
					break;
				case DTYPE_FLOAT64:
					kernel_softmax_generic<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output), ml::getPointer<double>(input), first_dim,
							last_dim);
					break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

