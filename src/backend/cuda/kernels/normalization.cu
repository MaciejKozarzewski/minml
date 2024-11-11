/*
 * batchnorm.cu
 *
 *  Created on: Jan 5, 2023
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

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors2;

	__device__ float get_mean(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[idx];
	}
	__device__ float get_stddev(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return std::sqrt(ptr[last_dim + idx] + 1.0e-6f);
	}
	__device__ float get_gamma(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[2 * last_dim + idx];
	}
	__device__ float get_beta(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[3 * last_dim + idx];
	}

	template<typename T>
	__device__ T square(T x)
	{
		return x * x;
	}
	template<typename T>
	__device__ T cube(T x)
	{
		return x * x * x;
	}

	/*
	 * Welford's online algorithm for calculating mean and variance
	 */
	template<typename T>
	struct AvgVarStats
	{
			T samples = static_cast<T>(0);
			T M = static_cast<T>(0); // mean
			T M2 = static_cast<T>(0); // variance

			AvgVarStats() = default;
			__device__ AvgVarStats(T n, T mean, T var) :
					samples(n),
					M(mean),
					M2(var)
			{
			}
			__device__ void add(T x) noexcept
			{
				samples += static_cast<T>(1);
				const T delta = x - M;
				M += delta / samples;
				M2 += delta * (x - M);
			}
			__device__ T get_average() const noexcept
			{
				return M;
			}
			__device__ T get_variance() const noexcept
			{
				assert(samples >= static_cast<T>(2));
				return M2 / (samples - static_cast<T>(1));
			}
			__device__ T get_stddev() const noexcept
			{
				return std::sqrt(static_cast<T>(1.0e-6) + get_variance());
			}
			__device__ void merge_with(const AvgVarStats<T> &other) noexcept
			{
				if (other.samples == static_cast<T>(0))
					return;
				if (this->samples == static_cast<T>(0))
				{
					this->samples = other.samples;
					this->M = other.M;
					this->M2 = other.M2;
				}
				else
				{
					const T total_samples = this->samples + other.samples;
					const T total_M = (this->samples * this->M + other.samples * other.M) / total_samples;
					const T total_M2 = this->M2 + other.M2 + square(this->M - other.M) * (this->samples * other.samples) / total_samples;
					this->samples = total_samples;
					this->M = total_M;
					this->M2 = total_M2;
				}
			}
	};

	__device__ void combine_stats(AvgVarStats<float> *stats)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				stats[threadIdx.y * 32 + threadIdx.x].merge_with(stats[(i + threadIdx.y) * 32 + threadIdx.x]);
			__syncthreads();
		}
	}
	__global__ void kernel_batchnorm_forward_avg_var_1(AvgVarStats<float> *workspace, const float *input, int first_dim, int last_dim)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ AvgVarStats<float> shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

		const int tid = blockIdx.x * 32 + threadIdx.x;

		AvgVarStats<float> thread_stat;
		if (tid < last_dim)
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				thread_stat.add(input[i * last_dim + tid]);

		shared_stats[threadIdx.y * 32 + threadIdx.x] = thread_stat;
		__syncthreads();

		combine_stats(shared_stats);
		if (threadIdx.y == 0 && tid < last_dim)
			workspace[blockIdx.y * last_dim + tid] = shared_stats[threadIdx.x];
	}
	__global__ void kernel_batchnorm_forward_avg_var_2(AvgVarStats<float> *running_stat, const AvgVarStats<float> *workspace, int first_dim,
			int last_dim)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ AvgVarStats<float> shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

		const int tid = blockIdx.x * 32 + threadIdx.x;

		AvgVarStats<float> thread_stat;
		if (tid < last_dim)
			for (int i = threadIdx.y; i < first_dim; i += 32)
				thread_stat.merge_with(workspace[i * last_dim + tid]);

		shared_stats[threadIdx.y * 32 + threadIdx.x] = thread_stat;
		__syncthreads();

		combine_stats(shared_stats);
		if (threadIdx.y == 0 && tid < last_dim)
			running_stat[tid] = shared_stats[threadIdx.x];
	}

	__global__ void kernel_batchnorm_forward(const float *weights, const float *input, float *output, const AvgVarStats<float> *running_stats,
			int2 shape, ml::mlActivationType_t act)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int first_dim = shape.x;
		const int last_dim = shape.y;
		if (tid < last_dim)
		{
			const float mean = running_stats[tid].get_average();
			const float stddev = running_stats[tid].get_stddev();
			const float gamma = get_gamma(weights, tid, last_dim);
			const float beta = get_beta(weights, tid, last_dim);

			const float scale = gamma / stddev;
			const float shift = -mean * scale + beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				float tmp = input[i * last_dim + tid] * scale + shift;
				if (act == ml::ACTIVATION_RELU)
					tmp = max(0.0f, tmp);
				if (act == ml::ACTIVATION_TANH)
					tmp = tanh(tmp);
				if (act == ml::ACTIVATION_SIGMOID)
					tmp = 1.0f / (1.0f + exp(-tmp));
				output[i * last_dim + tid] = tmp;
			}
		}
	}

	__global__ void kernel_batchnorm_inference(const float *weights, const float *input, float *output, int2 shape, ml::mlActivationType_t act)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < shape.y)
		{
			const float mean = get_mean(weights, tid, shape.y);
			const float stddev = get_stddev(weights, tid, shape.y);
			const float gamma = get_gamma(weights, tid, shape.y);
			const float beta = get_beta(weights, tid, shape.y);

			const float scale = gamma / stddev;
			const float shift = -mean * scale + beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < shape.x; i += gridDim.y * blockDim.y)
			{
				float tmp = input[i * shape.y + tid] * scale + shift;
				if (act == ml::ACTIVATION_RELU)
					tmp = max(0.0f, tmp);
				if (act == ml::ACTIVATION_TANH)
					tmp = tanh(tmp);
				if (act == ml::ACTIVATION_SIGMOID)
					tmp = 1.0f / (1.0f + exp(-tmp));
				output[i * shape.y + tid] = tmp;
			}
		}
	}

	__device__ void reduce_add_32x32_dual(float *ptr1, float *ptr2)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
			{
				ptr1[threadIdx.y * 32 + threadIdx.x] += ptr1[(i + threadIdx.y) * 32 + threadIdx.x];
				ptr2[threadIdx.y * 32 + threadIdx.x] += ptr2[(i + threadIdx.y) * 32 + threadIdx.x];
			}
			__syncthreads();
		}
	}
	__global__ void kernel_batchnorm_backward_delta_1(float *workspace, const float *input, const float *output, float *gradient_next,
			const AvgVarStats<float> *running_stats, int2 shape, ml::mlActivationType_t act)
	{
		__shared__ float d_sigma[32 * 32];
		__shared__ float d_mu[32 * 32];
		const int tid = blockIdx.x * 32 + threadIdx.x;

		float d_sigma_acc = 0.0f, d_mu_acc = 0.0f;
		if (tid < shape.y)
		{
			const float mean = running_stats[tid].get_average();
			const float stddev = running_stats[tid].get_stddev();
			for (int i = 32 * blockIdx.y + threadIdx.y; i < shape.x; i += 32 * gridDim.y)
			{
				const int tmp_idx = i * shape.y + tid;
				if (act == ml::ACTIVATION_RELU && output[tmp_idx] == 0.0f)
					gradient_next[tmp_idx] = 0.0f;
				if (act == ml::ACTIVATION_TANH)
					gradient_next[tmp_idx] *= (1.0f - output[tmp_idx]) * (1.0f + output[tmp_idx]);
				if (act == ml::ACTIVATION_SIGMOID)
					gradient_next[tmp_idx] *= output[tmp_idx] * (1.0f - output[tmp_idx]);
				d_sigma_acc += gradient_next[tmp_idx] * (input[tmp_idx] - mean) / stddev;
				d_mu_acc += gradient_next[tmp_idx];
			}
		}
		d_sigma[threadIdx.y * 32 + threadIdx.x] = d_sigma_acc;
		d_mu[threadIdx.y * 32 + threadIdx.x] = d_mu_acc;

		__syncthreads();
		reduce_add_32x32_dual(d_sigma, d_mu);
		if (threadIdx.y == 0 && tid < shape.y)
		{
			workspace[2 * blockIdx.y * shape.y + tid] = d_sigma[threadIdx.x];
			workspace[(2 * blockIdx.y + 1) * shape.y + tid] = d_mu[threadIdx.x];
		}
	}
	__global__ void kernel_batchnorm_backward_delta_2(float *workspace, int2 shape)
	{
		__shared__ float storage_d_sigma[32 * 32];
		__shared__ float storage_d_mu[32 * 32];
		const int tid = blockIdx.x * 32 + threadIdx.x;
		float d_sigma = 0.0f, d_mu = 0.0f;
		if (tid < shape.y)
			for (int i = 32 * blockIdx.y + threadIdx.y; i < shape.x; i += 32 * gridDim.y)
			{
				d_sigma += workspace[i * 2 * shape.y + tid];
				d_mu += workspace[(i * 2 + 1) * shape.y + tid];
			}
		storage_d_sigma[threadIdx.y * 32 + threadIdx.x] = d_sigma;
		storage_d_mu[threadIdx.y * 32 + threadIdx.x] = d_mu;

		__syncthreads();
		reduce_add_32x32_dual(storage_d_sigma, storage_d_mu);
		if (threadIdx.y == 0 && tid < shape.y)
		{
			workspace[tid] = storage_d_sigma[threadIdx.x];
			workspace[shape.y + tid] = storage_d_mu[threadIdx.x];
		}
	}

	__global__ void kernel_batchnorm_backward_1(const float *workspace, const float *input, float *gradient_prev, const float *gradient_next,
			const float *weights, float *weight_update, const AvgVarStats<float> *running_stats, int2 shape)
	{
		// avg, stddev, gamma, d_sigma, d_mu
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < shape.y)
		{
			const float mean = running_stats[tid].get_average();
			const float stddev = running_stats[tid].get_stddev();
			const float gamma = get_gamma(weights, tid, shape.y);

			float d_sigma = workspace[tid];
			float d_mu = workspace[shape.y + tid];
			if (blockIdx.y == 0 && threadIdx.y == 0)
			{ // only single line can update this
				weight_update[2 * shape.y + tid] += d_sigma; // gamma
				weight_update[3 * shape.y + tid] += d_mu; // beta
			}

			d_sigma = -gamma / stddev * d_sigma / static_cast<float>(shape.x - 1);
			d_mu = -gamma / stddev * d_mu / static_cast<float>(shape.x);
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < shape.x; i += gridDim.y * blockDim.y)
				gradient_prev[i * shape.y + tid] = gamma / stddev * gradient_next[i * shape.y + tid]
						+ d_sigma * (input[i * shape.y + tid] - mean) / stddev + d_mu;
		}
	}

	__global__ void kernel_batchnorm_update(const AvgVarStats<float> *running_stat, float *weights, int first_dim, int last_dim, bool use_gamma,
			bool use_beta)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < last_dim)
		{
			AvgVarStats<float> stats;
			for (int i = 0; i < first_dim; i++)
				stats.merge_with(running_stat[i * last_dim + tid]);
			weights[0 * last_dim + tid] = stats.get_average();
			weights[1 * last_dim + tid] = stats.get_variance();

			if (!use_gamma)
				weights[2 * last_dim + tid] = 1.0f; // gamma
			if (!use_beta)
				weights[3 * last_dim + tid] = 0.0f; // beta
		}
	}

	__global__ void kernel_fold_batchnorm(int first_dim, int last_dim, float *layer_weights, float *layer_bias, const float *batchnorm_weights)
	{
		const float mean = get_mean(batchnorm_weights, blockIdx.x, first_dim);
		const float stddev = get_stddev(batchnorm_weights, blockIdx.x, first_dim);
		const float gamma = get_gamma(batchnorm_weights, blockIdx.x, first_dim);
		const float beta = get_beta(batchnorm_weights, blockIdx.x, first_dim);

		const float scale = gamma / stddev;
		const float shift = -mean * scale + beta;
		for (int i = threadIdx.x; i < last_dim; i += blockDim.x)
			layer_weights[blockIdx.x * last_dim + i] *= scale;

		if (threadIdx.x == 0)
			layer_bias[blockIdx.x] = layer_bias[blockIdx.x] * scale + shift;
	}

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
	template<>
	__device__ half get_inv_stddev(half variance, int N, float epsilon)
	{
		return static_cast<half>(1.0f) / hsqrt(static_cast<half>(epsilon) + variance / static_cast<half>(N - 1));
	}

	template<int N, typename T, typename U>
	__device__ void vector_copy(T *dst, const U *src)
	{
		store_vec(dst, load_vec<U, N>(src));
	}

	template<typename T, int N>
	__launch_bounds__(128, 8)
	__global__ void kernel_layernorm_forward(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(last_dim % 4 == 0);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ AvgVarStats<float> stats[4];
		if (threadIdx.x < 4)
			stats[threadIdx.x] = AvgVarStats<float>();
		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			AvgVarStats<float> local_stats;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				local_stats.merge_with(get_stats(in));
				in.store(workspace + j);
			}
			__syncthreads();
			for (int k = 16; k >= 1; k /= 2)
			{
				const float n = __shfl_xor_sync(0xffffffff, local_stats.samples, k);
				const float avg = __shfl_xor_sync(0xffffffff, local_stats.M, k);
				const float var = __shfl_xor_sync(0xffffffff, local_stats.M2, k);

				const AvgVarStats<float> other(n, avg, var);
				local_stats.merge_with(other);
			}
			if (threadIdx.x % 32 == 0)
				stats[threadIdx.x / 32] = local_stats;
			__syncthreads();
			if (threadIdx.x == 0)
			{
				local_stats = AvgVarStats<float>();
				for (int k = 0; k < blockDim.x / 32; k++)
					local_stats.merge_with(stats[k]);
				stats[0] = local_stats;
			}
			__syncthreads();
			const float avg = stats[0].get_average();
			const float inv_stddev = 1.0f / stats[0].get_stddev();

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> gamma = load_vec<float, N>(weights + j);
				const vec<float, N> beta = load_vec<float, N>(bias + j);
				const vec<float, N> in = load_vec<float, N>(workspace + j);
				const vec<float, N> out = gamma * (in - avg) * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
			}
		}
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
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

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
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

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
	__global__ void kernel_layernorm_backward(const float *input, float *gradient_prev, float *gradient_next, const float *weights,
			float *weights_update, float *bias_update, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32);

		extern __shared__ char shared_array[];

		float *shared_weights_update = reinterpret_cast<float*>(shared_array);
		float *shared_bias_update = shared_weights_update + last_dim;
		float *shared_weights = shared_bias_update + last_dim;

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int j = N * tid; j < last_dim; j += N * blockDim.x * blockDim.y)
		{
			const vec<float, N> zero(0.0f);
			zero.store(shared_weights_update + j);
			zero.store(shared_bias_update + j);
			const vec<float, N> w(weights + j);
			w.store(shared_weights + j);
		}
		__syncthreads();

		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			AvgVarStats<float> local_stats;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> tmp(input + i * last_dim + j);
				local_stats.merge_with(get_stats(tmp));
			}
			for (int k = 16; k >= 1; k /= 2)
			{
				const float n = __shfl_xor_sync(0xffffffff, local_stats.samples, k);
				const float avg = __shfl_xor_sync(0xffffffff, local_stats.M, k);
				const float var = __shfl_xor_sync(0xffffffff, local_stats.M2, k);

				const AvgVarStats<float> other(n, avg, var);
				local_stats.merge_with(other);
			}
			const float avg = local_stats.get_average();
			const float inv_stddev = 1.0f / local_stats.get_stddev();

			float d_sigma = 0.0f;
			float d_mu = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const int idx = i * last_dim + j;
				const vec<float, N> in(input + idx);
				const vec<float, N> grad(gradient_next + idx);
				const vec<float, N> gamma(shared_weights + j);
				const vec<float, N> x = (in - avg) * inv_stddev;

				d_sigma -= horizontal_add(grad * x * gamma);
				d_mu -= horizontal_add(grad * gamma);

				atomic_add(shared_weights_update + j, grad * x);
				atomic_add(shared_bias_update + j, grad);
			}
			for (int k = 16; k >= 1; k /= 2)
			{
				d_sigma += __shfl_xor_sync(0xffffffff, d_sigma, k);
				d_mu += __shfl_xor_sync(0xffffffff, d_mu, k);
			}
			d_sigma *= inv_stddev / (last_dim - 1);
			d_mu *= inv_stddev / last_dim;

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const int idx = i * last_dim + j;
				const vec<float, N> in(input + idx);
				const vec<float, N> grad(gradient_next + idx);
				const vec<float, N> gamma(shared_weights + j);
				const vec<float, N> tmp = (gamma * grad + d_sigma * (in - avg)) * inv_stddev + d_mu;
				tmp.store(gradient_prev + idx);
			}
		}
		__syncthreads();
		for (int j = N * tid; j < last_dim; j += N * blockDim.x * blockDim.y)
		{
			const vec<float, N> dw(shared_weights_update + j);
			const vec<float, N> db(shared_bias_update + j);
			dw.store(weights_update + blockIdx.y * last_dim + j);
			db.store(bias_update + blockIdx.y * last_dim + j);
		}

//		assert(blockDim.x <= 128);
//		__shared__ float workspace[3 * 128];
//
//		for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//		{
//			weights_update[blockIdx.x * last_dim + j] = 0.0f;
//			bias_update[blockIdx.x * last_dim + j] = 0.0f;
//		}
//		__syncthreads();
//
//		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			// first recalculate input mean and variance
//			AvgVarStats<float> thread_stats;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				thread_stats.add(input[i * last_dim + j]);
//			AvgVarStats<float> *stats = reinterpret_cast<AvgVarStats<float>*>(workspace);
//			stats[threadIdx.x] = thread_stats;
//			__syncthreads();
//			combine_stats_1D(stats);
//			const float avg = stats[0].get_average();
//			const float stddev = stats[0].get_stddev();
//			__syncthreads();
//
//			float thread_d_sigma = 0.0f;
//			float thread_d_mu = 0.0f;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const int idx = i * last_dim + j;
//				const float gamma = weights[j];
//				const float x = (input[idx] - avg) / stddev;
//
//				thread_d_sigma -= gradient_next[idx] * x * gamma / stddev;
//				thread_d_mu -= gradient_next[idx] * gamma / stddev;
//
//				weights_update[blockIdx.x * last_dim + j] += gradient_next[idx] * x;
//				bias_update[blockIdx.x * last_dim + j] += gradient_next[idx];
//			}
//			workspace[threadIdx.x] = thread_d_sigma / (last_dim - 1);
//			workspace[threadIdx.x + 128] = thread_d_mu / last_dim;
//			__syncthreads();
//			combine_stats_1D(workspace, workspace + 128);
//			const float d_sigma = workspace[0];
//			const float d_mu = workspace[0 + 128];
//			__syncthreads();
//
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const int idx = i * last_dim + j;
//				const float gamma = weights[j];
//				gradient_prev[idx] = gamma / stddev * gradient_next[idx] + d_sigma * (input[idx] - avg) / stddev + d_mu;
//			}
//		}
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
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		vec<float, N> thread_weights_update = 0.0f;
		vec<float, N> thread_bias_update = 0.0f;

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
//			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
//			{
//				const vec<float, N> in = load_vec<float, N>(shared_input + j);
//				var += horizontal_add(square(in - avg));
//			}
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

//			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
//			{
//				const vec<float, N> in = load_vec<float, N>(shared_input + j);
//				const vec<float, N> grad = load_vec<float, N>(shared_gradient + j);
//				const vec<float, N> gamma = load_vec<float, N>(shared_weights + j);
//				const vec<float, N> x = (in - avg) * inv_stddev;
//
//				d_sigma -= horizontal_add(grad * x * gamma);
//				d_mu -= horizontal_add(grad * gamma);
//				store_vec(shared_input + j, x);
//
//				const vec<float, N> dw = load_vec<float, N>(shared_weights_update + j);
//				const vec<float, N> db = load_vec<float, N>(shared_bias_update + j);
//				store_vec(shared_weights_update + j, dw + grad * x);
//				store_vec(shared_bias_update + j, db + grad);
//			}
			d_sigma = cg::reduce(tile, d_sigma, cg::plus<float>()) * inv_stddev / (last_dim - 1);
			d_mu = cg::reduce(tile, d_mu, cg::plus<float>()) * inv_stddev / last_dim;

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				const vec<float, N> grad = load_vec<float, N>(shared_gradient + j);
//				const vec<float, N> gamma = load_vec<float, N>(shared_weights + j);
//				const vec<float, N> tmp = gamma * grad * inv_stddev + d_sigma * in + d_mu;
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

	template<typename T, int N>
	__global__ void kernel_rmsnorm_forward(const T *input, T *output, const T *weights, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(last_dim % N == 0);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ float stats[4];
		if (threadIdx.x < 4)
			stats[threadIdx.x] = 0.0f;
		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float local_sum_squares = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in(input + i * last_dim + j);
				local_sum_squares += horizontal_add(square(in));
				in.store(workspace + j);
			}
			__syncthreads();
			for (int k = 16; k >= 1; k /= 2)
				local_sum_squares += __shfl_xor_sync(0xffffffff, local_sum_squares, k);
			if (threadIdx.x % 32 == 0)
				stats[threadIdx.x / 32] = local_sum_squares;
			__syncthreads();
			if (threadIdx.x == 0)
			{
				local_sum_squares = 0.0f;
				for (int k = 0; k < blockDim.x / 32; k++)
					local_sum_squares += stats[k];
				stats[0] = local_sum_squares;
			}
			__syncthreads();

			local_sum_squares = stats[0];
			const float rms = std::sqrt(local_sum_squares / last_dim);
			const float inv_rms = 1.0f / (1.0e-6f + rms);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> gamma(weights + j);
				const vec<float, N> in(workspace + j);
				const vec<float, N> out = gamma * in * inv_rms;
				out.store(output + i * last_dim + j);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_rmsnorm_forward_v2(const T *input, T *output, const T *weights, int first_dim, int last_dim)
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
	void cuda_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		int2 dim { first_dim, last_dim };

		dim3 blockDim(32, 8);
		dim3 gridDim((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		kernel_batchnorm_inference<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<float>(weights), getPointer<float>(input),
				getPointer<float>(output), dim, act);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights, void *running_stats,
			int running_stat_idx, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		AvgVarStats<float> *workspace = cuda::Context::getWorkspace<AvgVarStats<float>>(context);
		const int workspace_first_dim = std::min((size_t) 256, cuda::Context::getWorkspaceSize(context) / (sizeof(AvgVarStats<float> ) * last_dim));
		assert(workspace_first_dim > 0);

		AvgVarStats<float> *running_stats_ptr = getPointer<AvgVarStats<float>>(running_stats) + running_stat_idx * last_dim;

		dim3 blockDim(32, 32);
		dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);

		int2 shape1 { first_dim, last_dim };
		dim3 gridDim2(gridDim1.x);
//		int2 shape2 { workspace_first_dim, last_dim };
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_batchnorm_forward_avg_var_1<<<gridDim1, blockDim, 0,stream >>>(workspace, getPointer<float>(input), first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
		kernel_batchnorm_forward_avg_var_2<<<gridDim2, blockDim, 0, stream>>>(running_stats_ptr, workspace, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim3(32, 8);
		dim3 gridDim3((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		kernel_batchnorm_forward<<<gridDim3, blockDim3, 0, stream>>>(getPointer<float>(weights), getPointer<float>(input), getPointer<float>(output),
				running_stats_ptr, shape1, act);

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
			void *gradient_next, const void *weights, void *weights_update, const void *running_stats, int running_stat_idx, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		float *workspace = cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) 256, cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		const AvgVarStats<float> *running_stats_ptr = getPointer<AvgVarStats<float>>(running_stats) + running_stat_idx * last_dim;

		dim3 blockDim(32, 32);
		dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);

		int2 shape1 { first_dim, last_dim };
		dim3 gridDim2(gridDim1.x);
		int2 shape2 { workspace_first_dim, last_dim };

		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_batchnorm_backward_delta_1<<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<float>(input), getPointer<float>(output),
				getPointer<float>(gradient_next), running_stats_ptr, shape1, act);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_batchnorm_backward_delta_2<<<gridDim2, blockDim, 0, stream>>>(workspace, shape2);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim3(32, 8);
		dim3 gridDim3((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		kernel_batchnorm_backward_1<<<gridDim3, blockDim3, 0, stream>>>(workspace, getPointer<float>(input), getPointer<float>(gradient_prev),
				getPointer<float>(gradient_next), getPointer<float>(weights), getPointer<float>(weights_update), running_stats_ptr, shape1);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape) / 3;

		dim3 blockDim(256);
		dim3 gridDim(std::max(1u, (last_dim + blockDim.x - 1) / blockDim.x));
		kernel_batchnorm_update<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<AvgVarStats<float>>(running_stat),
				getPointer<float>(weights), first_dim, last_dim, use_gamma, use_beta);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = volume_without_first_dim(shape);
		dim3 blockDim(256);
		dim3 gridDim(first_dim);

		kernel_fold_batchnorm<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(first_dim, last_dim, getPointer<float>(layer_weights),
				getPointer<float>(layer_bias), getPointer<float>(batchnorm_weights));
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights,
			const void *bias, const void *ext)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(256);
		dim3 gridDim(std::min(512, first_dim));

		cudaStream_t stream = cuda::Context::getStream(context);

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

//		float *workspace = cuda::Context::getWorkspace<float>(context);
//		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 2048),
//				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));

//		dim3 blockDim(128);
//		dim3 gridDim(workspace_first_dim);
//
//		cudaStream_t stream = cuda::Context::getStream(context);
//
//		float *partial_weights_update = workspace;
//		float *partial_bias_update = workspace + workspace_first_dim * last_dim;
//
//		kernel_layernorm_backward<1><<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
//				getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
//		assert(cudaGetLastError() == cudaSuccess);

		cudaStream_t stream = cuda::Context::getStream(context);

//		dim3 blockDim(32, 8);
//
//		float *workspace = cuda::Context::getWorkspace<float>(context);
//		const int workspace_first_dim = std::min((size_t) std::min((first_dim + (int) blockDim.y - 1) / (int) blockDim.y, 128),
//				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));
//		float *partial_weights_update = workspace;
//		float *partial_bias_update = workspace + workspace_first_dim * last_dim;
//		const int shared_mem = sizeof(float) * 3 * last_dim;
//
//		dim3 gridDim(1, workspace_first_dim);
//		if (last_dim % 4 == 0)
//		{
//			kernel_layernorm_backward<4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
//					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
//		}
//		else
//		{
//			kernel_layernorm_backward<1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
//					getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
//		}

		dim3 blockDim(256);

		float *workspace = cuda::Context::getWorkspace<float>(context);
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

	void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(256);
		dim3 gridDim(std::min(1024, first_dim));

		cudaStream_t stream = cuda::Context::getStream(context);

		const bool use_gamma = (weights != nullptr);

		const int shared_mem = sizeof(float) * (1 + use_gamma) * last_dim;
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_rmsnorm_forward_v2<half, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input), getPointer<half>(output),
							getPointer<half>(weights), first_dim, last_dim);
				else
					kernel_rmsnorm_forward_v2<half, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<half>(input), getPointer<half>(output),
							getPointer<half>(weights), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_rmsnorm_forward_v2<float, 4> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input),
							getPointer<float>(output), getPointer<float>(weights), first_dim, last_dim);
				else
					kernel_rmsnorm_forward_v2<float, 1> <<<gridDim, blockDim, shared_mem, stream >>>(getPointer<float>(input),
							getPointer<float>(output), getPointer<float>(weights), first_dim, last_dim);
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

		float *workspace = cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min((first_dim + (int) blockDim.y - 1) / (int) blockDim.y, 128),
				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 gridDim(1, workspace_first_dim);

		cudaStream_t stream = cuda::Context::getStream(context);

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

