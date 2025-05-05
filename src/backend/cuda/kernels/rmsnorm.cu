/*
 * rmsnorm.cu
 *
 *  Created on: Mar 28, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"
#include "../helpers/indexers.cuh"

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

	template<typename T, int N>
	__global__ void kernel_rmsnorm_forward(const T *input, T *output, const T *weights, int first_dim, int last_dim)
	{
		extern __shared__ char shared_array[];

		float *shared_input = reinterpret_cast<float*>(shared_array);
		float *shared_weights = shared_input + last_dim;

		const bool use_gamma = (weights != nullptr);

		if (use_gamma)
		{
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				vector_copy<N>(shared_weights + j, weights + j);
			__syncthreads();
		}

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float sum_squares = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				sum_squares += horizontal_add(square(in));
				store_vec(shared_input + j, in);
			}
			sum_squares = cg::reduce(tile, sum_squares, cg::plus<float>());
			const float rms = std::sqrt(sum_squares / last_dim);
			const float inv_rms = 1.0f / (1.0e-6f + rms);

			if (use_gamma)
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const vec<float, N> gamma = load_vec<float, N>(weights + j);
					const vec<float, N> in = load_vec<float, N>(shared_input + j);
					const vec<float, N> out = gamma * in * inv_rms;
					store_vec(output + i * last_dim + j, out);
				}
			}
			else
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const vec<float, N> in = load_vec<float, N>(shared_input + j);
					store_vec(output + i * last_dim + j, in * inv_rms);
				}
			}
		}
	}
	template<int N>
	__global__ void kernel_rmsnorm_backward(const float *input, float *gradient_prev, float *gradient_next, const float *weights,
			float *weights_update, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32);

		extern __shared__ char shared_array[];

		float *shared_update = reinterpret_cast<float*>(shared_array);
		float *shared_weights = shared_update + last_dim;

		const bool use_gamma = (weights != nullptr);

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (use_gamma)
		{
			for (int j = N * tid; j < last_dim; j += N * blockDim.x * blockDim.y)
			{
				const vec<float, N> w(weights + j);
				w.store(shared_weights + j);
				const vec<float, N> zero(0.0f);
				zero.store(shared_update + j);
			}
			__syncthreads();
		}

		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			float sum_squares = 0.0f;
			float sum = 0.0f;
			if (use_gamma)
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const int idx = i * last_dim + j;
					const vec<float, N> in(input + idx);
					const vec<float, N> grad(gradient_next + idx);
					const vec<float, N> gamma(shared_weights + j);
					sum_squares += horizontal_add(square(in));
					sum += horizontal_add(in * grad * gamma);
				}
			}
			else
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const int idx = i * last_dim + j;
					const vec<float, N> in(input + idx);
					const vec<float, N> grad(gradient_next + idx);
					sum_squares += horizontal_add(square(in));
					sum += horizontal_add(in * grad);
				}
			}
			for (int k = 16; k >= 1; k /= 2)
			{
				sum_squares += __shfl_xor_sync(0xffffffff, sum_squares, k);
				sum += __shfl_xor_sync(0xffffffff, sum, k);
			}

			const float rms = std::sqrt(sum_squares / last_dim);
			const float inv_rms = 1.0f / (1.0e-6f + rms);
			const float mult = 1.0f / (last_dim * cube(rms));
			sum_squares *= mult;
			sum *= mult;
			if (use_gamma)
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const int idx = i * last_dim + j;
					const vec<float, N> in(input + idx);
					const vec<float, N> grad(gradient_next + idx);
					const vec<float, N> gamma(shared_weights + j);
					const vec<float, N> out = in * inv_rms;

					atomic_add(shared_update + j, grad * out);
					const vec<float, N> tmp = gamma * grad * sum_squares - in * sum;
					tmp.store(gradient_prev + idx);
				}
			}
			else
			{
				for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				{
					const int idx = i * last_dim + j;
					const vec<float, N> in(input + idx);
					const vec<float, N> grad(gradient_next + idx);

					const vec<float, N> tmp = grad * sum_squares - in * sum;
					tmp.store(gradient_prev + idx);
				}
			}
		}
		if (use_gamma)
		{
			__syncthreads();
			for (int j = N * tid; j < last_dim; j += N * blockDim.x * blockDim.y)
			{
				const vec<float, N> w(shared_update + j);
				w.store(weights_update + blockIdx.y * last_dim + j);
			}
		}
	}

	__global__ void kernel_reduce_first_dim(float *dst, const float *src, int first_dim, int last_dim)
	{
		__shared__ float workspace[32][32 + 1];

		const int last_dim_idx = 32 * blockIdx.x + threadIdx.x;
		if (last_dim_idx < last_dim)
		{
			float local_sum = 0.0f;
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				local_sum += src[i * last_dim + last_dim_idx];
			workspace[threadIdx.y][threadIdx.x] = local_sum;
		}
		__syncthreads();
		float local_sum = workspace[threadIdx.x][threadIdx.y];

		for (int k = 16; k >= 1; k /= 2)
			local_sum += __shfl_xor_sync(0xffffffff, local_sum, k);
		__syncthreads();
		if (threadIdx.x == 0)
			workspace[0][threadIdx.y] = local_sum;
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
			dst[last_dim_idx] += workspace[0][threadIdx.x];
	}
}

namespace ml
{

	void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(256);
		dim3 gridDim(std::min(1024, first_dim));

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		const bool use_gamma = (weights != nullptr);

		const int shared_mem = sizeof(float) * (1 + use_gamma) * last_dim;
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_rmsnorm_forward<half, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input), getPointer<half>(output),
							getPointer<half>(weights), first_dim, last_dim);
				else
					kernel_rmsnorm_forward<half, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input), getPointer<half>(output),
							getPointer<half>(weights), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_rmsnorm_forward<float, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(output),
							getPointer<float>(weights), first_dim, last_dim);
				else
					kernel_rmsnorm_forward<float, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(output),
							getPointer<float>(weights), first_dim, last_dim);
				break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_rmsnorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next, const void *weights,
			void *weights_update)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(32, 4);

		float *workspace = ml::cuda_backend::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min((first_dim + (int) blockDim.y - 1) / (int) blockDim.y, 128),
				ml::cuda_backend::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 gridDim(1, workspace_first_dim);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		float *partial_weights_update = workspace;
		const bool use_gamma = (weights != nullptr);
		const int shared_mem = sizeof(float) * (1 + use_gamma) * last_dim;

		if (last_dim % 4 == 0)
		{
			kernel_rmsnorm_backward<4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, first_dim, last_dim);
		}
		else
		{
			kernel_rmsnorm_backward<1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, first_dim, last_dim);
		}
		assert(cudaGetLastError() == cudaSuccess);

		if (use_gamma)
		{
			dim3 blockDim2(32, 32);
			dim3 gridDim2((last_dim + 31) / 32);
			kernel_reduce_first_dim<<<gridDim2, blockDim2, 0, stream >>>( getPointer<float>(weights_update), partial_weights_update,
					workspace_first_dim, last_dim);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

