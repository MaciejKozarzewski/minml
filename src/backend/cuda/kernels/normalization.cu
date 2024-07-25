/*
 * batchnorm.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"

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

			d_sigma = -gamma / stddev * d_sigma / static_cast<float>(shape.x);
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

	struct __builtin_align__(16) vec4f
	{
			float x0, x1, x2, x3;

	};
	__device__ vec4f operator+(const vec4f &lhs, const vec4f &rhs)
	{
		vec4f result;
		result.x0 = lhs.x0 + rhs.x0;
		result.x1 = lhs.x1 + rhs.x1;
		result.x2 = lhs.x2 + rhs.x2;
		result.x3 = lhs.x3 + rhs.x3;
		return result;
	}
	__device__ vec4f operator-(const vec4f &lhs, const vec4f &rhs)
	{
		vec4f result;
		result.x0 = lhs.x0 - rhs.x0;
		result.x1 = lhs.x1 - rhs.x1;
		result.x2 = lhs.x2 - rhs.x2;
		result.x3 = lhs.x3 - rhs.x3;
		return result;
	}
	__device__ vec4f operator-(const vec4f &lhs, float rhs)
	{
		vec4f result;
		result.x0 = lhs.x0 - rhs;
		result.x1 = lhs.x1 - rhs;
		result.x2 = lhs.x2 - rhs;
		result.x3 = lhs.x3 - rhs;
		return result;
	}
	__device__ vec4f operator*(const vec4f &lhs, const vec4f &rhs)
	{
		vec4f result;
		result.x0 = lhs.x0 * rhs.x0;
		result.x1 = lhs.x1 * rhs.x1;
		result.x2 = lhs.x2 * rhs.x2;
		result.x3 = lhs.x3 * rhs.x3;
		return result;
	}
	__device__ vec4f operator*(const vec4f &lhs, float rhs)
	{
		vec4f result;
		result.x0 = lhs.x0 * rhs;
		result.x1 = lhs.x1 * rhs;
		result.x2 = lhs.x2 * rhs;
		result.x3 = lhs.x3 * rhs;
		return result;
	}
	struct __builtin_align__(8) vec4h
	{
			half x0, x1, x2, x3;
	};

	__device__ vec4f load_vec(const float *ptr)
	{
		return reinterpret_cast<const vec4f*>(ptr)[0];
	}
	__device__ vec4f load_vec(const half *ptr)
	{
		const vec4h tmp = reinterpret_cast<const vec4h*>(ptr)[0];
		vec4f result;
		result.x0 = tmp.x0;
		result.x1 = tmp.x1;
		result.x2 = tmp.x2;
		result.x3 = tmp.x3;
		return result;
	}
	__device__ void store_vec(float *ptr, const vec4f &x)
	{
		reinterpret_cast<vec4f*>(ptr)[0] = x;
	}
	__device__ void store_vec(half *ptr, const vec4f &x)
	{
		vec4h tmp;
		tmp.x0 = x.x0;
		tmp.x1 = x.x1;
		tmp.x2 = x.x2;
		tmp.x3 = x.x3;
		reinterpret_cast<vec4h*>(ptr)[0] = tmp;
	}
	__device__ float reduce_sum(const vec4f &v)
	{
		return v.x0 + v.x1 + v.x2 + v.x3;
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
//		assert(last_dim <= 1024);
//		assert(blockDim.x == 256);
//		float workspace[8];
//		__shared__ cg::block_tile_memory<256> btm;
//		cg::thread_block thb = cg::this_thread_block(btm);
//		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);
//
//		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			int counter = 0;
//			float local_avg = 0.0f;
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				float tmp = input[i * last_dim + j];
//				if (ext != nullptr)
//					tmp += ext[i * last_dim + j];
//				local_avg += tmp;
//				workspace[counter] = tmp;
//				counter++;
//			}
//			const float mean = cg::reduce(tile, local_avg, cg::plus<float>()) / last_dim;
//
//			counter = 0;
//			float local_var = 0.0f;
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				float tmp = workspace[counter] - mean;
//				local_var += tmp * tmp;
//				counter++;
//			}
//			const float var = cg::reduce(tile, local_var, cg::plus<float>()) / (last_dim - 1);
//			const float inv_stddev = 1.0f / std::sqrt(1.0e-6f + var);
//
//			counter = 0;
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				const float gamma = weights[j];
//				const float beta = bias[j];
//				output[i * last_dim + j] = gamma * (workspace[counter] - mean) * inv_stddev + beta;
//				counter++;
//			}
//		}

//		assert(last_dim <= 1024);
//		assert(blockDim.x == 64);
//		__shared__ float workspace[512];
//		__shared__ cg::block_tile_memory<64> btm;
//		cg::thread_block thb = cg::this_thread_block(btm);
//		cg::thread_block_tile<64> tile = cg::tiled_partition<64>(thb);
//
//		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			float local_avg = 0.0f;
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				float tmp = input[i * last_dim + j];
//				if (ext != nullptr)
//					tmp += ext[i * last_dim + j];
//				local_avg += tmp;
//				workspace[j] = tmp;
//			}
//			const float mean = cg::reduce(tile, local_avg, cg::plus<float>()) / last_dim;
//
//			float local_var = 0.0f;
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				float tmp = workspace[j] - mean;
//				local_var += tmp * tmp;
//			}
//			const float var = cg::reduce(tile, local_var, cg::plus<float>()) / (last_dim - 1);
//			const float inv_stddev = 1.0f / std::sqrt(1.0e-6f + var);
//
//			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
//			{
//				const float gamma = 1.0f; // weights[j];
//				const float beta = 0.0f; //bias[j];
//				output[i * last_dim + j] = gamma * (workspace[j] - mean) * inv_stddev + beta;
//			}
//		}

//		assert(last_dim <= 1024);
//		assert(last_dim % 4 == 0);
//		assert(blockDim.x <= 256);
//		__shared__ cg::block_tile_memory<256> btm;
//		cg::thread_block thb = cg::this_thread_block(btm);
//		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);
//
//		const vec4f gamma = load_vec(weights + 4 * tile.thread_rank());
//		const vec4f beta = load_vec(bias + 4 * tile.thread_rank());
//
//		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			const int idx = i * last_dim + 4 * tile.thread_rank();
//			vec4f workspace = load_vec(input + idx);
//			if (ext != nullptr)
//				workspace = workspace + load_vec(ext + idx);
//
//			const float local_mean = reduce_sum(workspace);
//			const float mean = cg::reduce(tile, local_mean, cg::plus<float>()) / last_dim;
//
//			workspace = workspace - mean;
//			float local_var = reduce_sum(workspace * workspace);
//
//			const float var = cg::reduce(tile, local_var, cg::plus<float>()) / (last_dim - 1);
//			const float inv_stddev = 1.0f / std::sqrt(1.0e-6f + var);
//
//			const vec4f out = gamma * workspace * inv_stddev + beta;
//			store_vec(output + idx, out);
//		}

//		assert(last_dim <= 1024);
//		assert(blockDim.x <= 128);
//		__shared__ float workspace[1024];
//		__shared__ AvgVarStats<float> stats[128];
//
//		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			AvgVarStats<float> local_stats;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				workspace[j] = input[i * last_dim + j];
//				if (ext != nullptr)
//					workspace[j] += ext[i * last_dim + j];
//				local_stats.add(workspace[j]);
//			}
//			stats[threadIdx.x] = local_stats;
//			__syncthreads();
//			combine_stats_1D(stats);
//			const float avg = stats[0].get_average();
//			const float stddev = stats[0].get_stddev();
//			__syncthreads();
//
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const float gamma = weights[j];
//				const float beta = bias[j];
//				output[i * last_dim + j] = gamma * (workspace[j] - avg) / stddev + beta;
//			}
//		}

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
				const vec4f tmp = load_vec(input + i * last_dim + j);
				local_stats.merge_with(get_stats(tmp));
				store_vec(workspace + j, tmp);
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
				const vec4f gamma = load_vec(weights + j);
				const vec4f beta = load_vec(bias + j);
				const vec4f tmp = load_vec(workspace + j);
				const vec4f out = gamma * (tmp - avg) * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
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
			workspace[threadIdx.x] = thread_d_sigma / last_dim;
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

		std::cout << first_dim << " " << last_dim << " : " << workspace_first_dim << '\n';

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

} /* namespace ml */

