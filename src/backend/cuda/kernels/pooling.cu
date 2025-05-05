/*
 * pooling.cu
 *
 *  Created on: Apr 21, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../helpers/misc.cuh"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;
	using namespace ml;

	template<typename T, int N>
	__global__ void kernel_average_pooling_forward(float beta, T *output, float alpha, const T *input, int batch_size, int height, int width,
			int channels, int pool_size)
	{
		assert(channels % N == 0);
		assert(blockDim.x == 32);
		assert(blockDim.y <= 8);

		__shared__ vec<T, N> workspace[8][32];

		const Indexer<4> input_indexer(batch_size, height, width, channels);
		const Indexer<4> output_indexer(batch_size, (height + pool_size - 1) / pool_size, (width + pool_size - 1) / pool_size, channels);

		const int origin_h = blockIdx.y * pool_size;
		const int origin_w = blockIdx.z * pool_size;
		const int local_count = min(pool_size, height - origin_h) * min(pool_size, width - origin_w);

		for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
		{
			vec<T, N> local_sum(0.0f);
			for (int i = threadIdx.y; i < pool_size * pool_size; i += blockDim.y)
			{
				const int h = origin_h + i / pool_size;
				const int w = origin_w + i % pool_size;
				if (is_inside(h, w, height, width))
					local_sum += vec<T, N>(input + input_indexer.at(blockIdx.x, h, w, c));
			}
			workspace[threadIdx.y][threadIdx.x] = local_sum;
			__syncthreads();
			if (threadIdx.y == 0)
			{
				vec<T, N> final_sum(0.0f);
				for (int k = 0; k < blockDim.y; k++)
					final_sum += workspace[k][threadIdx.x];

				final_sum *= vec<T, N>(alpha / local_count);
				const int idx = output_indexer.at(blockIdx.x, blockIdx.y, blockIdx.z, c);
				if (beta != 0.0f)
					final_sum += vec<T, N>(beta) * vec<T, N>(output + idx);
				final_sum.store(output + idx);
			}
			__syncthreads();
		}
	}
	template<typename T, int N>
	__global__ void kernel_average_pooling_backward(float beta, T *gradient_prev, float alpha, const T *gradient_next, int batch_size, int height,
			int width, int channels, int pool_size)
	{
		assert(channels % N == 0);

		const Indexer<4> input_indexer(batch_size, height, width, channels);
		const Indexer<4> output_indexer(batch_size, (height + pool_size - 1) / pool_size, (width + pool_size - 1) / pool_size, channels);

		const int origin_h = blockIdx.y * pool_size;
		const int origin_w = blockIdx.z * pool_size;
		const int local_count = min(pool_size, height - origin_h) * min(pool_size, width - origin_w);

		for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
		{
			vec<T, N> grad = vec<T, N>(gradient_next + output_indexer.at(blockIdx.x, blockIdx.y, blockIdx.z, c)) * vec<T, N>(alpha / local_count);
			for (int i = threadIdx.y; i < pool_size * pool_size; i += blockDim.y)
			{
				const int h = origin_h + i / pool_size;
				const int w = origin_w + i % pool_size;
				if (is_inside(h, w, height, width))
				{
					const int idx = input_indexer.at(blockIdx.x, h, w, c);
					vec<T, N> dx = grad;
					if (beta != 0.0f)
						dx += vec<T, N>(beta) * vec<T, N>(gradient_prev + idx);
					dx.store(gradient_prev + idx);
				}
			}
		}
	}

}

namespace ml
{
	void cuda_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y, int size)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int channels = x.dim[3];

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, std::min(size * size, 8));
		dim3 gridDim(batch_size, (height + size - 1) / size, (width + size - 1) / size);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<half, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x), batch_size,
							height, width, channels, size);
				else
					kernel_average_pooling_forward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x), batch_size,
							height, width, channels, size);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, height, width, channels, size);
				else
					kernel_average_pooling_forward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, height, width, channels, size);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_average_pooling_forward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						batch_size, height, width, channels, size);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx, int size)
	{
		assert(dx.rank == 4);
		assert(dy.rank == 4);
		const int batch_size = dx.dim[0];
		const int height = dx.dim[1];
		const int width = dx.dim[2];
		const int channels = dx.dim[3];

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, std::min(size * size, 8));
		dim3 gridDim(batch_size, (height + size - 1) / size, (width + size - 1) / size);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_backward<half, 4> <<<gridDim,blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							batch_size, height, width, channels, size);
				else
					kernel_average_pooling_backward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							batch_size, height, width, channels, size);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_backward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(dx), alpha, data<float>(dy),
							batch_size, height, width, channels, size);
				else
					kernel_average_pooling_backward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(dx), alpha, data<float>(dy),
							batch_size, height, width, channels, size);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_average_pooling_backward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<double>(dx), alpha, data<double>(dy),
						batch_size, height, width, channels, size);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
}

