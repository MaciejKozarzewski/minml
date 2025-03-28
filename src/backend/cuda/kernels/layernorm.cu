/*
 * layernorm.cu
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
#include "../helpers/AvgVarStats.cuh"

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
	__device__ AvgVarStats<T> get_stats(const vec<T, N> &v)
	{
		const T mean = horizontal_add(v) / static_cast<T>(N);
		const T var = horizontal_add(square(v - mean));
		return AvgVarStats<T>(static_cast<T>(v.size()), mean, var);
	}
	template<typename T>
	__device__ AvgVarStats<T> get_stats(const vec<T, 1> &v)
	{
		AvgVarStats<T> result;
		result.add(v.x0);
		return result;
	}

	template<typename T>
	__device__ T get_inv_stddev(T variance, int N, float epsilon);

	template<>
	__device__ float get_inv_stddev(float variance, int N, float epsilon)
	{
		return 1.0f / std::sqrt(epsilon + variance / (N - 1));
	}

	template<typename T, int N>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_v2(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 256);

		extern __shared__ char shared_array[];

		float *shared_input = reinterpret_cast<float*>(shared_array);
		float *shared_weights = shared_input + last_dim;
		float *shared_bias = shared_weights + last_dim;

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			vector_copy<N>(shared_weights + j, weights + j);
			vector_copy<N>(shared_bias + j, bias + j);
		}
		__syncthreads();

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 256 > tile = cg::tiled_partition<256>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + j, in);
			}
			avg = cg::reduce(tile, avg, cg::plus<float>()) / static_cast<float>(last_dim);

			float var = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				var += horizontal_add(square(in - avg));
			}
			const float inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<float>()), last_dim, 1.0e-6f);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> gamma = load_vec<float, N>(shared_weights + j);
				const vec<float, N> beta = load_vec<float, N>(shared_bias + j);
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				const vec<float, N> out = gamma * (in - avg) * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
			}
		}
	}
	template<typename T, int N>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_v3(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 256);

		extern __shared__ char shared_array[];

		T *shared_weights = reinterpret_cast<T*>(shared_array);
		T *shared_bias = shared_weights + last_dim;

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			const vec<T, N> w = load_vec<T, N>(weights + j);
			const vec<T, N> b = load_vec<T, N>(bias + j);
			store_vec(shared_weights + j, w);
			store_vec(shared_bias + j, b);
		}
		__syncthreads();

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 256 > tile = cg::tiled_partition<256>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			vec<T, N> in;

			T avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				in = load_vec<T, N>(input + i * last_dim + j);
				avg += horizontal_add(in);
			}
			avg = cg::reduce(tile, avg, cg::plus<T>()) / static_cast<T>(last_dim);

			T var = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				in -= avg;
				var += horizontal_add(square(in));
			}
			const T inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<T>()), last_dim, 1.0e-6f);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> gamma = load_vec<T, N>(shared_weights + j);
				const vec<T, N> beta = load_vec<T, N>(shared_bias + j);
				const vec<T, N> out = gamma * in * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
			}
		}
	}
	template<int N>
	__global__ void kernel_layernorm_backward_v2(const float *input, float *gradient_prev, float *gradient_next, const float *weights,
			float *weights_update, float *bias_update, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);

		extern __shared__ char shared_array[];

		float *shared_input = reinterpret_cast<float*>(shared_array);
		float *shared_gradient = shared_input + last_dim;
		float *shared_weights = shared_gradient + last_dim;
		float *shared_weights_update = shared_weights + last_dim;
		float *shared_bias_update = shared_weights_update + last_dim;

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 256 > tile = cg::tiled_partition<256>(thb);

		vec<float, N> thread_weights_update(0.0f);
		vec<float, N> thread_bias_update(0.0f);

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			const vec<float, N> zero(0.0f);
			zero.store(shared_weights_update + j);
			zero.store(shared_bias_update + j);
			vector_copy<N>(shared_weights + j, weights + j);
		}

		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				const vec<float, N> grad = load_vec<float, N>(gradient_next + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + j, in);
				store_vec(shared_gradient + j, grad);
			}
			avg = cg::reduce(tile, avg, cg::plus<float>()) / last_dim;

			float var = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
				var += square(shared_input[j] - avg);
			const float inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<float>()), last_dim, 1.0e-6f);

			float d_sigma = 0.0f;
			float d_mu = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const float in = (shared_input[j] - avg) * inv_stddev;
				const float grad = shared_gradient[j];
				const float gamma = shared_weights[j];

				d_sigma -= grad * in * gamma;
				d_mu -= grad * gamma;
				shared_weights_update[j] += grad * in;
				shared_bias_update[j] += grad;

				shared_input[j] = in;
				shared_gradient[j] = grad * gamma;
			}

			d_sigma = cg::reduce(tile, d_sigma, cg::plus<float>()) * inv_stddev / (last_dim - 1);
			d_mu = cg::reduce(tile, d_mu, cg::plus<float>()) * inv_stddev / last_dim;

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				const vec<float, N> grad = load_vec<float, N>(shared_gradient + j);
				const vec<float, N> tmp = grad * inv_stddev + d_sigma * in + d_mu;
				store_vec(gradient_prev + i * last_dim + j, tmp);
			}
		}
		__syncthreads();
		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			vector_copy<N>(weights_update + blockIdx.x * last_dim + j, shared_weights_update + j);
			vector_copy<N>(bias_update + blockIdx.x * last_dim + j, shared_bias_update + j);
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

	void cuda_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights,
			const void *bias, const void *ext)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(256);
		dim3 gridDim(std::min(512, first_dim));

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				const int shared_mem = sizeof(float) * 3 * last_dim;
				if (last_dim % 4 == 0)
					kernel_layernorm_forward_v2<half, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input),
							getPointer<half>(output), getPointer<half>(weights), getPointer<half>(bias), getPointer<half>(ext), first_dim, last_dim);
				else
					kernel_layernorm_forward_v2<half, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input),
							getPointer<half>(output), getPointer<half>(weights), getPointer<half>(bias), getPointer<half>(ext), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const int shared_mem = sizeof(float) * 2 * last_dim;
				if (last_dim % 4 == 0)
					kernel_layernorm_forward_v3<float, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input),
							getPointer<float>(output), getPointer<float>(weights), getPointer<float>(bias), getPointer<float>(ext), first_dim,
							last_dim);
				else
					kernel_layernorm_forward_v3<float, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input),
							getPointer<float>(output), getPointer<float>(weights), getPointer<float>(bias), getPointer<float>(ext), first_dim,
							last_dim);
				break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
			const void *weights, void *weights_update, void *bias_update)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		dim3 blockDim(256);

		float *workspace = ml::cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 512),
				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));
		float *partial_weights_update = workspace;
		float *partial_bias_update = workspace + workspace_first_dim * last_dim;
		const int shared_mem = sizeof(float) * 5 * last_dim;

		dim3 gridDim(workspace_first_dim);
		if (last_dim % 4 == 0)
		{
			kernel_layernorm_backward_v2<4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
		}
		else
		{
			kernel_layernorm_backward_v2<1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_reduce_first_dim<<<gridDim2, blockDim2, 0, stream >>>(getPointer<float>(weights_update), partial_weights_update, workspace_first_dim,
				last_dim);
		kernel_reduce_first_dim<<<gridDim2, blockDim2, 0, stream >>>(getPointer<float>(bias_update), partial_bias_update, workspace_first_dim,
				last_dim);

		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

