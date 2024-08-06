/*
 * batchnorm.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec4f.cuh"

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

			__device__ AvgVarStats() = default;
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

	__device__ void combine_stats_1D(AvgVarStats<float> *stats)
	{
		for (int j = blockDim.x / 2; j >= 1; j /= 2)
		{
			if (threadIdx.x < j)
				stats[threadIdx.x].merge_with(stats[threadIdx.x + j]);
			__syncthreads();
		}
	}
	__device__ void combine_stats_1D(float *stats1, float *stats2)
	{
		for (int j = blockDim.x / 2; j >= 1; j /= 2)
		{
			if (threadIdx.x < j)
			{
				stats1[threadIdx.x] += stats1[threadIdx.x + j];
				stats2[threadIdx.x] += stats2[threadIdx.x + j];
			}
			__syncthreads();
		}
	}

	__device__ AvgVarStats<float> get_stats(const vec4f &v)
	{
		const float mean = (v.x0 + v.x1 + v.x2 + v.x3) / 4.0f;
		const float var = square(v.x0 - mean) + square(v.x1 - mean) + square(v.x2 - mean) + square(v.x3 - mean);
		return AvgVarStats<float>(4.0f, mean, var);
	}

	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward(const float *input, float *output, const float *weights, const float *bias, const float *ext,
			int first_dim, int last_dim)
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
			for (int j = 4 * threadIdx.x; j < last_dim; j += 4 * blockDim.x)
			{
				const vec4f tmp(input + i * last_dim + j);
				local_stats.merge_with(get_stats(tmp));
				tmp.store(workspace + j);
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

			for (int j = 4 * threadIdx.x; j < last_dim; j += 4 * blockDim.x)
			{
				const vec4f gamma(weights + j);
				const vec4f beta(bias + j);
				const vec4f tmp(workspace + j);
				const vec4f out = gamma * (tmp - avg) * inv_stddev + beta;
				out.store(output + i * last_dim + j);
			}
		}
	}
	__global__ void kernel_layernorm_backward(const float *input, float *gradient_prev, float *gradient_next, const float *weights,
			float *weights_update, float *bias_update, int first_dim, int last_dim)
	{
		assert(blockDim.x <= 128);
		__shared__ float workspace[3 * 128];

		for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
		{
			weights_update[blockIdx.x * last_dim + j] = 0.0f;
			bias_update[blockIdx.x * last_dim + j] = 0.0f;
		}
		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			// first recalculate input mean and variance
			AvgVarStats<float> thread_stats;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
				thread_stats.add(input[i * last_dim + j]);
			AvgVarStats<float> *stats = reinterpret_cast<AvgVarStats<float>*>(workspace);
			stats[threadIdx.x] = thread_stats;
			__syncthreads();
			combine_stats_1D(stats);
			const float avg = stats[0].get_average();
			const float stddev = stats[0].get_stddev();
			__syncthreads();

			float thread_d_sigma = 0.0f;
			float thread_d_mu = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const int idx = i * last_dim + j;
				const float gamma = weights[j];
				const float x = (input[idx] - avg) / stddev;

				thread_d_sigma -= gradient_next[idx] * x * gamma / stddev;
				thread_d_mu -= gradient_next[idx] * gamma / stddev;

				weights_update[blockIdx.x * last_dim + j] += gradient_next[idx] * x;
				bias_update[blockIdx.x * last_dim + j] += gradient_next[idx];
			}
			workspace[threadIdx.x] = thread_d_sigma / (last_dim - 1);
			workspace[threadIdx.x + 128] = thread_d_mu / last_dim;
			__syncthreads();
			combine_stats_1D(workspace, workspace + 128);
			const float d_sigma = workspace[0];
			const float d_mu = workspace[0 + 128];
			__syncthreads();

			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const int idx = i * last_dim + j;
				const float gamma = weights[j];
				gradient_prev[idx] = gamma / stddev * gradient_next[idx] + d_sigma * (input[idx] - avg) / stddev + d_mu;
			}
		}
	}
	__global__ void kernel_layernorm_update(const float *partial_weights_upadte, const float *partial_bias_upadte, float *weights_update,
			float *bias_update, int first_dim, int last_dim)
	{
		__shared__ float storage_w[32 * 32];
		__shared__ float storage_b[32 * 32];
		const int tid = blockIdx.x * 32 + threadIdx.x;
		float thread_w = 0.0f;
		float thread_b = 0.0f;
		if (tid < last_dim)
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
			{
				thread_w += partial_weights_upadte[i * last_dim + tid];
				thread_b += partial_bias_upadte[i * last_dim + tid];
			}
		storage_w[threadIdx.y * 32 + threadIdx.x] = thread_w;
		storage_b[threadIdx.y * 32 + threadIdx.x] = thread_b;

		__syncthreads();
		reduce_add_32x32_dual(storage_w, storage_b);
		if (threadIdx.y == 0 && tid < last_dim)
		{
			weights_update[tid] += storage_w[threadIdx.x];
			bias_update[tid] += storage_b[threadIdx.x];
		}
	}

	__global__ void kernel_rmsnorm_forward(const float *input, float *output, const float *weights, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(last_dim % 4 == 0);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ float stats[4];
		if (threadIdx.x < 4)
			stats[threadIdx.x] = 0.0f;
		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float local_sum_squares = 0.0f;
			for (int j = 4 * threadIdx.x; j < last_dim; j += 4 * blockDim.x)
			{
				const vec4f tmp(input + i * last_dim + j);
				local_sum_squares += horizontal_add(tmp * tmp);
				tmp.store(workspace + j);
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

			const float rms = std::sqrt(local_sum_squares / last_dim);
			const float inv_rms = 1.0f / (1.0e-6f + rms);

			for (int j = 4 * threadIdx.x; j < last_dim; j += 4 * blockDim.x)
			{
				const vec4f gamma(weights + j);
				const vec4f tmp(workspace + j);
				const vec4f out = gamma * tmp * inv_rms;
				out.store(output + i * last_dim + j);
			}
		}
	}
	__global__ void kernel_rmsnorm_backward(const float *input, float *gradient_prev, float *gradient_next, const float *weights,
			float *weights_update, int first_dim, int last_dim)
	{
		assert(blockDim.x == 128);
		__shared__ float workspace[2][4];

		for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			weights_update[blockIdx.x * last_dim + j] = 0.0f;
		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float local_sum_squares = 0.0f;
			float local_sum = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const int idx = i * last_dim + j;
				const float in = input[idx];
				const float grad = gradient_next[idx];
				const float gamma = weights[j];
				local_sum_squares += square(in);
				local_sum += in * grad * gamma;
			}
			__syncthreads();
			for (int k = 16; k >= 1; k /= 2)
			{
				local_sum_squares += __shfl_xor_sync(0xffffffff, local_sum_squares, k);
				local_sum += __shfl_xor_sync(0xffffffff, local_sum, k);
			}
			if (threadIdx.x % 32 == 0)
			{
				workspace[0][threadIdx.x / 32] = local_sum_squares;
				workspace[1][threadIdx.x / 32] = local_sum;
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				local_sum_squares = 0.0f;
				local_sum = 0.0f;
				for (int k = 0; k < blockDim.x / 32; k++)
				{
					local_sum_squares += workspace[0][k];
					local_sum += workspace[1][k];
				}
				workspace[0][0] = local_sum_squares;
				workspace[1][0] = local_sum;
			}
			__syncthreads();
			const float sum_squares = workspace[0][0];
			const float sum = workspace[1][0];

			const float rms = std::sqrt(sum_squares / last_dim);
			const float inv_rms = 1.0f / (1.0e-6f + rms);
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const int idx = i * last_dim + j;
				const float gamma = weights[j];
				const float in = input[idx];
				const float grad = gradient_next[idx];
				const float out = in * inv_rms;

				weights_update[blockIdx.x * last_dim + j] += gradient_next[idx] * out;
				gradient_prev[idx] = (gamma * grad * sum_squares - in * sum) / (last_dim * cube(rms));
			}
		}
	}
	__global__ void kernel_rmsnorm_update(const float *partial_weights_upadte, float *weights_update, int first_dim, int last_dim)
	{
		__shared__ float storage_w[32 * 32];
		const int tid = blockIdx.x * 32 + threadIdx.x;
		float thread_w = 0.0f;
		if (tid < last_dim)
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				thread_w += partial_weights_upadte[i * last_dim + tid];
		storage_w[threadIdx.y * 32 + threadIdx.x] = thread_w;

		__syncthreads();
		for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
				storage_w[threadIdx.y * 32 + threadIdx.x] += storage_w[(i + threadIdx.y) * 32 + threadIdx.x];
			__syncthreads();
		}
		if (threadIdx.y == 0 && tid < last_dim)
			weights_update[tid] += storage_w[threadIdx.x];
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

		dim3 blockDim(128);
		dim3 gridDim(std::min(2048, first_dim));

		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_layernorm_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(output), getPointer<float>(weights),
				getPointer<float>(bias), getPointer<float>(ext), first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
			const void *weights, void *weights_update, void *bias_update)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		float *workspace = cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 512),
				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));

		dim3 blockDim(128);
		dim3 gridDim(workspace_first_dim);

		cudaStream_t stream = cuda::Context::getStream(context);

		float *partial_weights_update = workspace;
		float *partial_bias_update = workspace + workspace_first_dim * last_dim;

		kernel_layernorm_backward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
				getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_layernorm_update<<<gridDim2, blockDim2, 0, stream >>>(partial_weights_update, partial_bias_update, getPointer<float>(weights_update),
				getPointer<float>(bias_update), workspace_first_dim, last_dim);

		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(128);
		dim3 gridDim(std::min(2048, first_dim));

		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_rmsnorm_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(output), getPointer<float>(weights),
				first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_rmsnorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next, const void *weights,
			void *weights_update)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		float *workspace = cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 512),
				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 blockDim(128);
		dim3 gridDim(workspace_first_dim);

		cudaStream_t stream = cuda::Context::getStream(context);

		float *partial_weights_update = workspace;

		kernel_rmsnorm_backward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
				getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_rmsnorm_update<<<gridDim2, blockDim2, 0, stream >>>(partial_weights_update, getPointer<float>(weights_update), workspace_first_dim,
				last_dim);

		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

