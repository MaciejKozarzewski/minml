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

	template<typename T, typename U>
	__global__ void kernel_softmax_3_channels(T *output, const T *input, int first_dim)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < first_dim)
		{
			U x0 = static_cast<U>(input[idx * 3 + 0]);
			U x1 = static_cast<U>(input[idx * 3 + 1]);
			U x2 = static_cast<U>(input[idx * 3 + 2]);

			const U max_value = max(x0, max(x1, x2));
			x0 = exp(x0 - max_value);
			x1 = exp(x1 - max_value);
			x2 = exp(x2 - max_value);

			const U inv_sum = 1.0f / (x0 + x1 + x2);

			output[idx * 3 + 0] = static_cast<T>(x0 * inv_sum);
			output[idx * 3 + 1] = static_cast<T>(x1 * inv_sum);
			output[idx * 3 + 2] = static_cast<T>(x2 * inv_sum);
		}
	}

	template<typename T, typename U>
	__global__ void kernel_softmax_forward_generic(T *output, const T *input, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);
		__shared__ U workspace[1024];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<128> tile = cg::tiled_partition<128>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			U max_value = -1e+32f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = static_cast<U>(input[i * last_dim + j]);
				max_value = max(max_value, workspace[j]);
			}
			const U shift = cg::reduce(tile, max_value, cg::greater<U>());

			U partial_sum = 0.0f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = exp(workspace[j] - shift);
				partial_sum += workspace[j];
			}
			const U inv_sum = 1.0f / cg::reduce(tile, partial_sum, cg::plus<U>());
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
				output[i * last_dim + j] = static_cast<T>(workspace[j] * inv_sum);
		}
	}
	template<typename T, typename U>
	__global__ void kernel_softmax_backward_generic(float alpha, const T *output, const T *gradient_next, float beta, T *gradient_prev, int first_dim,
			int last_dim)
	{
		assert(last_dim <= 512);
		assert(blockDim.x == 128);
		__shared__ U tmp_y[512];
		__shared__ U tmp_dy[512];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<128> tile = cg::tiled_partition<128>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			U tmp = 0;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				const U y = static_cast<U>(output[i * last_dim + j]);
				const U dy = static_cast<U>(gradient_next[i * last_dim + j]);
				tmp_y[j] = y;
				tmp_dy[j] = dy;
				tmp += dy * y;
			}
			const U partial_sum = cg::reduce(tile, tmp, cg::plus<U>());

			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				const U y = tmp_y[j];
				const U dy = tmp_dy[j];
				U dx = static_cast<U>(alpha) * y * (dy - tmp);
				if (beta != 0.0f)
					dx += static_cast<U>(beta) * static_cast<U>(gradient_prev[i * last_dim + j]);
				gradient_prev[i * last_dim + j] = static_cast<T>(dx);
			}
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
					kernel_softmax_3_channels<half, float> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
							ml::getPointer<half>(input), first_dim);
					break;
				case DTYPE_FLOAT32:
					kernel_softmax_3_channels<float, float> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), first_dim);
					break;
				case DTYPE_FLOAT64:
					kernel_softmax_3_channels<double, double> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output),
							ml::getPointer<double>(input), first_dim);
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
					kernel_softmax_forward_generic<half, float> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
							ml::getPointer<half>(input), first_dim, last_dim);
					break;
				case DTYPE_FLOAT32:
					kernel_softmax_forward_generic<float, float> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), first_dim, last_dim);
					break;
				case DTYPE_FLOAT64:
					kernel_softmax_forward_generic<double, double> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output),
							ml::getPointer<double>(input), first_dim, last_dim);
					break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_softmax_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t y, float beta, mlTensor_t dx)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		assert(dy.rank == 2);
		const int first_dim = get_first_dim(dy);
		const int last_dim = get_last_dim(dy);

		dim3 blockDim(128);
		dim3 gridDim(std::min(1024, first_dim));
		switch (dy.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_softmax_backward_generic<half, float> <<<gridDim, blockDim, 0, stream>>>(alpha, data<half>(y), data<half>(dy), beta,
						data<half>(dx), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_softmax_backward_generic<float, float> <<<gridDim, blockDim, 0, stream>>>(alpha, data<float>(y), data<float>(dy), beta,
						data<float>(dx), first_dim, last_dim);
				break;
			case DTYPE_FLOAT64:
				kernel_softmax_backward_generic<double, double> <<<gridDim, blockDim, 0, stream>>>(alpha, data<double>(y), data<double>(dy), beta,
						data<double>(dx), first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

