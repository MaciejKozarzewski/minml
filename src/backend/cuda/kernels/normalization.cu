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

	template<typename T>
	__device__ float get_mean(const T *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[idx];
	}
	template<typename T>
	__device__ float get_stddev(const T *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return sqrt(static_cast<float>(ptr[last_dim + idx]) + 1.0e-6f);
	}
	template<typename T>
	__device__ float get_gamma(const T *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[2 * last_dim + idx];
	}
	template<typename T>
	__device__ float get_beta(const T *ptr, int idx, int last_dim)
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
			T samples = get<T>(0.0f);
			T M = get<T>(0.0f); // mean
			T M2 = get<T>(0.0f); // variance

			AvgVarStats() = default;
			__device__ AvgVarStats(T n, T mean, T var) :
					samples(n),
					M(mean),
					M2(var)
			{
			}
			template<typename U>
			__device__ void add(U x) noexcept
			{
				samples += static_cast<T>(1.0f);
				const T delta = static_cast<T>(x) - M;
				M += delta / samples;
				M2 += delta * (static_cast<T>(x) - M);
			}
			__device__ T get_average() const noexcept
			{
				return M;
			}
			__device__ T get_variance() const noexcept
			{
				assert(samples >= static_cast<T>(2.0f));
				return M2 / (samples - static_cast<T>(1.0f));
			}
			__device__ T get_stddev() const noexcept
			{
				return std::sqrt(static_cast<T>(1.0e-6f) + get_variance());
			}
			__device__ void merge_with(const AvgVarStats<T> &other) noexcept
			{
				if (other.samples == static_cast<T>(0.0f))
					return;
				if (this->samples == static_cast<T>(0.0f))
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
	template<typename T>
	__global__ void kernel_batchnorm_forward_avg_var_1(AvgVarStats<float> *workspace, const T *input, int first_dim, int last_dim)
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
	template<typename T>
	__global__ void kernel_batchnorm_forward(float beta, T *output, float alpha, const T *input, const T *weights,
			const AvgVarStats<float> *running_stats, int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
				float tmp = static_cast<float>(input[i * last_dim + tid]) * scale + shift;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						tmp = ml::internal::sigmoid(tmp);
						break;
					case ml::ACTIVATION_TANH:
						tmp = ml::internal::tanh(tmp);
						break;
					case ml::ACTIVATION_RELU:
						tmp = ml::internal::relu(tmp);
						break;
				}
				tmp *= alpha;
				if (beta != 0.0f)
					tmp += beta * static_cast<float>(output[i * last_dim + tid]);
				output[i * last_dim + tid] = tmp;
			}
		}
	}

	template<typename T>
	__global__ void kernel_batchnorm_inference(float beta, T *output, float alpha, const T *input, const T *weights, int first_dim, int last_dim,
			ml::mlActivationType_t act)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < last_dim)
		{
			const float mean = get_mean(weights, tid, last_dim);
			const float stddev = get_stddev(weights, tid, last_dim);
			const float gamma = get_gamma(weights, tid, last_dim);
			const float beta = get_beta(weights, tid, last_dim);

			const float scale = gamma / stddev;
			const float shift = -mean * scale + beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				float tmp = static_cast<float>(input[i * last_dim + tid]) * scale + shift;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						tmp = ml::internal::sigmoid(tmp);
						break;
					case ml::ACTIVATION_TANH:
						tmp = ml::internal::tanh(tmp);
						break;
					case ml::ACTIVATION_RELU:
						tmp = ml::internal::relu(tmp);
						break;
				}
				tmp *= alpha;
				if (beta != 0.0f)
					tmp += beta * static_cast<float>(output[i * last_dim + tid]);
				output[i * last_dim + tid] = tmp;
			}
		}
	}

	/*
	 * new kernels
	 */
	template<typename T, int N>
	__global__ void kernel_batchnorm_inference(float beta, T *output, float alpha, const T *input, const T *weights, const T *stats, int first_dim,
			int last_dim, ml::mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		if (tid < last_dim)
		{
			const vec<T, N> _mean(stats + tid);
			const vec<T, N> _variance(stats + tid + last_dim);
			const vec<T, N> _gamma(weights + tid);
			const vec<T, N> _beta(weights + tid + last_dim);

			const vec<T, N> epsilon(1.0e-6f);

			const vec<T, N> scale = _gamma / vectors::sqrt(_variance + epsilon);
			const vec<T, N> shift = -_mean * scale + _beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int tmp_idx = i * last_dim + tid;
				vec<T, N> tmp(input + tmp_idx);
				tmp = tmp * scale + shift;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						tmp = vectors::sigmoid(tmp);
						break;
					case ml::ACTIVATION_TANH:
						tmp = vectors::tanh(tmp);
						break;
					case ml::ACTIVATION_RELU:
						tmp = vectors::relu(tmp);
						break;
				}
				tmp *= vec<T, N>(alpha);
				if (beta != 0.0f)
					tmp += vec<T, N>(beta) * vec<T, N>(output + tmp_idx);
				tmp.store(output + tmp_idx);
			}
		}
	}
	template<int N>
	__device__ void combine_stats_vec(AvgVarStats<float> *stats, const Indexer<3> &idx)
	{
		for (int i = blockDim.y / 2; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
			{
				for (int n = 0; n < N; n++)
					stats[idx.at(threadIdx.y, threadIdx.x, n)].merge_with(stats[idx.at(i + threadIdx.y, threadIdx.x, n)]);
			}
			__syncthreads();
		}
	}
	template<typename T, int N>
	__global__ void kernel_batchnorm_forward_avg_var_1_v2(AvgVarStats<float> *workspace, const T *input, int first_dim, int last_dim)
	{
		assert(blockDim.x == 32 && blockDim.y == 8);
		__shared__ AvgVarStats<float> shared_stats[8 * 32 * N]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		const Indexer<3> idx(blockDim.y, blockDim.x, N);

		AvgVarStats<float> thread_stat[N];
		for (int n = 0; n < N; n++)
			thread_stat[n] = AvgVarStats<float>();
		if (tid < last_dim)
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const vec<T, N> tmp(input + i * last_dim + tid);
				for (int n = 0; n < N; n++)
					thread_stat[n].add(tmp[n]);
			}

		for (int n = 0; n < N; n++)
			shared_stats[idx.at(threadIdx.y, threadIdx.x, n)] = thread_stat[n];
		__syncthreads();

		combine_stats_vec<N>(shared_stats, idx);
		if (threadIdx.y == 0 && tid < last_dim)
		{
			for (int n = 0; n < N; n++)
				workspace[blockIdx.y * last_dim + tid + n] = shared_stats[idx.at(0, threadIdx.x, n)];
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_forward_avg_var_2_v2(AvgVarStats<float> *running_stats, const AvgVarStats<float> *workspace, int first_dim,
			int last_dim)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ AvgVarStats<float> shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const Indexer<3> idx(blockDim.y, blockDim.x, 1);

		AvgVarStats<float> thread_stat;
		if (tid < last_dim)
			for (int i = threadIdx.y; i < first_dim; i += 32)
				thread_stat.merge_with(workspace[i * last_dim + tid]);

		shared_stats[idx.at(threadIdx.y, threadIdx.x, 0)] = thread_stat;
		__syncthreads();

		combine_stats_vec<1>(shared_stats, idx);
		if (threadIdx.y == 0 && tid < last_dim)
			running_stats[tid] = shared_stats[idx.at(0, threadIdx.x, 0)];
	}
	template<typename T>
	__global__ void kernel_calculate_avg_var(T *avg_var, const AvgVarStats<float> *running_stats, int last_dim)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < last_dim; i += gridDim.x * blockDim.x)
		{
			const AvgVarStats<float> tmp = running_stats[i];
			avg_var[i] = static_cast<T>(tmp.get_average());
			avg_var[i + last_dim] = static_cast<T>(tmp.get_variance());
		}
	}

	template<int N>
	__device__ void reduce_add_dual(float *ptr1, float *ptr2, const Indexer<3> &idx)
	{
		for (int i = blockDim.y / 2; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
			{
				for (int n = 0; n < N; n++)
				{
					ptr1[idx.at(threadIdx.y, threadIdx.x, n)] += ptr1[idx.at(i + threadIdx.y, threadIdx.x, n)];
					ptr2[idx.at(threadIdx.y, threadIdx.x, n)] += ptr2[idx.at(i + threadIdx.y, threadIdx.x, n)];
				}
			}
			__syncthreads();
		}
	}
	template<typename T, int N>
	__global__ void kernel_batchnorm_backward_delta_1_v2(float *workspace, const T *input, T *gradient_next, const T *weights, const T *stats,
			int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(blockDim.x == 32 && blockDim.y == 8);
		__shared__ float d_sigma[8 * 32 * N];
		__shared__ float d_mu[8 * 32 * N];
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);

		vec<float, N> d_sigma_acc(0.0f);
		vec<float, N> d_mu_acc(0.0f);
		if (tid < last_dim)
		{
			const vec<T, N> epsilon(1.0e-6f);
			const vec<T, N> _mean(stats + tid);
			const vec<T, N> _stddev = vectors::sqrt(vec<T, N>(stats + tid + last_dim) + epsilon);
			const vec<T, N> _gamma(weights + tid);
			const vec<T, N> _beta(weights + tid + last_dim);

			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = i * last_dim + tid;
				vec<T, N> grad(gradient_next + tmp_idx);
				const vec<T, N> inp(input + tmp_idx);
				const vec<T, N> hat_inp = (inp - _mean) / _stddev;
				vec<T, N> out = _gamma * hat_inp + _beta;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						out = vectors::sigmoid(out);
						grad *= out * (one<T, N>() - out);
						break;
					case ml::ACTIVATION_TANH:
						grad *= (one<T, N>() - square(vectors::tanh(out)));
						break;
					case ml::ACTIVATION_RELU:
						grad = select(out == zero<T, N>(), zero<T, N>(), grad);
						break;
				}
				d_sigma_acc += convert<float, T, N>(grad * hat_inp);
				d_mu_acc += convert<float, T, N>(grad);
				grad.store(gradient_next + tmp_idx);
			}
		}
		const Indexer<3> idx(blockDim.y, blockDim.x, N);
		for (int n = 0; n < N; n++)
		{
			d_sigma[idx.at(threadIdx.y, threadIdx.x, n)] = d_sigma_acc[n];
			d_mu[idx.at(threadIdx.y, threadIdx.x, n)] = d_mu_acc[n];
		}

		__syncthreads();
		reduce_add_dual<N>(d_sigma, d_mu, idx);
		if (threadIdx.y == 0 && tid < last_dim)
		{
			const vec<float, N> ds(d_sigma + N * threadIdx.x);
			const vec<float, N> dm(d_mu + N * threadIdx.x);
			ds.store(workspace + 2 * blockIdx.y * last_dim + tid);
			dm.store(workspace + (2 * blockIdx.y + 1) * last_dim + tid);
		}
	}
	__global__ void kernel_batchnorm_backward_delta_2_v2(float *workspace, int first_dim, int last_dim)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ float storage_d_sigma[32 * 32];
		__shared__ float storage_d_mu[32 * 32];
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		float d_sigma = 0.0f, d_mu = 0.0f;
		if (tid < last_dim)
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				d_sigma += workspace[i * 2 * last_dim + tid];
				d_mu += workspace[(i * 2 + 1) * last_dim + tid];
			}
		storage_d_sigma[threadIdx.y * blockDim.x + threadIdx.x] = d_sigma;
		storage_d_mu[threadIdx.y * blockDim.x + threadIdx.x] = d_mu;

		__syncthreads();
		const Indexer<3> idx(blockDim.y, blockDim.x, 1);
		reduce_add_dual<1>(storage_d_sigma, storage_d_mu, idx);
		if (threadIdx.y == 0 && tid < last_dim)
		{
			workspace[tid] = storage_d_sigma[threadIdx.x];
			workspace[last_dim + tid] = storage_d_mu[threadIdx.x];
		}
//		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//		{
//			for (int i = 0; i < last_dim; i++)
//				printf("%i : dsigma = %f, dmu = %f\n", i, workspace[i], workspace[last_dim + i]);
//		}
	}
	template<typename T, int N>
	__global__ void kernel_batchnorm_backward_v2(const float *d_sigma_mu, const T *input, T *gradient_prev, const T *gradient_next, const T *weights,
			const T *stats, T *weight_update, int first_dim, int last_dim, float alpha, float beta_dx, float beta_dw)
	{
		// avg, stddev, gamma, d_sigma, d_mu
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		if (tid < last_dim)
		{
			const vec<float, N> epsilon(1.0e-6f);
			const vec<float, N> _mean = load_vec<float, N>(stats + tid);
			const vec<float, N> _stddev = vectors::sqrt(load_vec<float, N>(stats + tid + last_dim) + epsilon);
			const vec<float, N> _gamma = load_vec<float, N>(weights + tid);

			vec<float, N> d_sigma = load_vec<float, N>(d_sigma_mu + tid);
			vec<float, N> d_mu = load_vec<float, N>(d_sigma_mu + tid + last_dim);
			if (blockIdx.y == 0 && threadIdx.y == 0)
			{ // only single line can update this
				vec<float, N> ds = d_sigma * vec<float, N>(alpha);
				vec<float, N> dm = d_mu * vec<float, N>(alpha);
				if (beta_dw != 0.0f)
				{
					ds += beta_dw * load_vec<float, N>(weight_update + 2 * last_dim + tid);
					dm += beta_dw * load_vec<float, N>(weight_update + 3 * last_dim + tid);
				}
				store_vec(weight_update + 2 * last_dim + tid, ds);
				store_vec(weight_update + 3 * last_dim + tid, dm);
			}

			const float m = static_cast<float>(first_dim);
			const vec<float, N> d_s = -_gamma / _stddev * d_sigma / (m - 1.0f);
			const vec<float, N> d_m = -_gamma / _stddev * d_mu / m;
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int tmp_idx = i * last_dim + tid;
				vec<float, N> tmp = _gamma / _stddev * load_vec<float, N>(gradient_next + tmp_idx)
						+ d_s * (load_vec<float, N>(input + tmp_idx) - _mean) / _stddev + d_m;
				tmp *= alpha;
				if (beta_dx != 0.0f)
					tmp += beta_dx * load_vec<float, N>(gradient_prev + tmp_idx);
				store_vec(gradient_prev + tmp_idx, tmp);
			}
		}
	}

	__device__ void reduce_add_dual(float *ptr1, float *ptr2)
	{
		for (int i = blockDim.y / 2; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
			{
				ptr1[threadIdx.y * blockDim.x + threadIdx.x] += ptr1[(i + threadIdx.y) * blockDim.x + threadIdx.x];
				ptr2[threadIdx.y * blockDim.x + threadIdx.x] += ptr2[(i + threadIdx.y) * blockDim.x + threadIdx.x];
			}
			__syncthreads();
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward_delta_1(float *workspace, const T *input, T *gradient_next, const AvgVarStats<float> *running_stats,
			const T *weights, int2 shape, ml::mlActivationType_t act)
	{
		__shared__ float d_sigma[32 * 8];
		__shared__ float d_mu[32 * 8];
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;

		float d_sigma_acc = 0.0f, d_mu_acc = 0.0f;
		if (tid < shape.y)
		{
			const float mean = running_stats[tid].get_average();
			const float stddev = running_stats[tid].get_stddev();
			const float gamma = get_gamma(weights, tid, shape.y);
			const float beta = get_beta(weights, tid, shape.y);
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < shape.x; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = i * shape.y + tid;
				float grad = gradient_next[tmp_idx];
				const float inp = input[tmp_idx];
				float out = gamma * (inp - mean) / stddev + beta;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
					{
						out = ml::internal::sigmoid(out);
						grad *= out * (1.0f - out);
						break;
					}
					case ml::ACTIVATION_TANH:
					{
						out = ml::internal::tanh(out);
						grad *= (1.0f - out) * (1.0f + out);
						break;
					}
					case ml::ACTIVATION_RELU:
					{
						grad = (out <= 0.0f) ? 0.0f : grad;
						break;
					}
				}
				d_sigma_acc += grad * (inp - mean) / stddev;
				d_mu_acc += grad;
				gradient_next[tmp_idx] = grad;
			}
		}
		d_sigma[threadIdx.y * blockDim.x + threadIdx.x] = d_sigma_acc;
		d_mu[threadIdx.y * blockDim.x + threadIdx.x] = d_mu_acc;

		__syncthreads();
		reduce_add_dual(d_sigma, d_mu);
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
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		float d_sigma = 0.0f, d_mu = 0.0f;
		if (tid < shape.y)
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < shape.x; i += blockDim.y * gridDim.y)
			{
				d_sigma += workspace[i * 2 * shape.y + tid];
				d_mu += workspace[(i * 2 + 1) * shape.y + tid];
			}
		storage_d_sigma[threadIdx.y * blockDim.x + threadIdx.x] = d_sigma;
		storage_d_mu[threadIdx.y * blockDim.x + threadIdx.x] = d_mu;

		__syncthreads();
		reduce_add_dual(storage_d_sigma, storage_d_mu);
		if (threadIdx.y == 0 && tid < shape.y)
		{
			workspace[tid] = storage_d_sigma[threadIdx.x];
			workspace[shape.y + tid] = storage_d_mu[threadIdx.x];
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward(const float *workspace, const T *input, T *gradient_prev, const T *gradient_next, const T *weights,
			T *weight_update, const AvgVarStats<float> *running_stats, int2 shape, float alpha, float beta_dx, float beta_dw)
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
				float ds = d_sigma * alpha;
				float dm = d_mu * alpha;
				if (beta_dw != 0.0f)
				{
					ds += beta_dw * static_cast<float>(weight_update[2 * shape.y + tid]);
					dm += beta_dw * static_cast<float>(weight_update[3 * shape.y + tid]);
				}
				weight_update[2 * shape.y + tid] = ds; // gamma
				weight_update[3 * shape.y + tid] = dm; // beta
			}

			d_sigma = -gamma / stddev * d_sigma / static_cast<float>(shape.x - 1);
			d_mu = -gamma / stddev * d_mu / static_cast<float>(shape.x);
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < shape.x; i += gridDim.y * blockDim.y)
			{
				float tmp = gamma / stddev * static_cast<float>(gradient_next[i * shape.y + tid])
						+ d_sigma * (static_cast<float>(input[i * shape.y + tid]) - mean) / stddev + d_mu;
				tmp *= alpha;
				if (beta_dx != 0.0f)
					tmp += beta_dx * static_cast<float>(gradient_prev[i * shape.y + tid]);
				gradient_prev[i * shape.y + tid] = tmp;
			}
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_update(const AvgVarStats<float> *running_stat, T *weights, int first_dim, int last_dim, bool use_gamma,
			bool use_beta)
	{
		__shared__ AvgVarStats<float> shared_stats[8 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < last_dim)
		{
			const Indexer<3> idx(blockDim.y, blockDim.x, 1);
			AvgVarStats<float> thread_stat;
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				thread_stat.merge_with(running_stat[i * last_dim + tid]);

			shared_stats[idx.at(threadIdx.y, threadIdx.x, 0)] = thread_stat;
			__syncthreads();

			combine_stats_vec<1>(shared_stats, idx);
			if (threadIdx.y == 0)
			{
				weights[0 * last_dim + tid] = static_cast<T>(shared_stats[idx.at(0, threadIdx.x, 0)].get_average());
				weights[1 * last_dim + tid] = static_cast<T>(shared_stats[idx.at(0, threadIdx.x, 0)].get_variance());

				if (!use_gamma)
					weights[2 * last_dim + tid] = static_cast<T>(1.0f); // gamma
				if (!use_beta)
					weights[3 * last_dim + tid] = static_cast<T>(0.0f); // beta
			}
		}
	}

	__global__ void kernel_fold_batchnorm_4D(int first_dim, int last_dim, float *layer_weights, float *layer_bias, const float *batchnorm_weights)
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
	__global__ void kernel_fold_batchnorm_3D(int first_dim, int last_dim, float *layer_weights, float *layer_bias, const float *batchnorm_weights)
	{
		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const float stddev = get_stddev(batchnorm_weights, j, last_dim);
				const float gamma = get_gamma(batchnorm_weights, j, last_dim);
				const float scale = gamma / stddev;
				layer_weights[i * last_dim + j] *= scale;

				if (i == 0)
				{
					const float mean = get_mean(batchnorm_weights, j, last_dim);
					const float beta = get_beta(batchnorm_weights, j, last_dim);
					const float shift = -mean * scale + beta;
					layer_bias[j] = layer_bias[j] * scale + shift;
				}
			}
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
//	template<>
//	__device__ half get_inv_stddev(half variance, int N, float epsilon)
//	{
//		return static_cast<half>(1.0f) / hsqrt(static_cast<half>(epsilon) + variance / static_cast<half>(N - 1));
//	}

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

	void calculate_avg_var(cudaStream_t stream, const AvgVarStats<float> *running_stats_ptr, void *avg_var, ml::mlDataType_t dtype, int last_dim)
	{
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
				kernel_calculate_avg_var<half> <<<1, 1024, 0, stream>>>(ml::getPointer<half>(avg_var), running_stats_ptr, last_dim);
				break;
			case ml::DTYPE_FLOAT32:
				kernel_calculate_avg_var<float> <<<1, 1024, 0, stream>>>(ml::getPointer<float>(avg_var), running_stats_ptr, last_dim);
				break;
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	void cuda_batchnorm_inference(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t stats, float beta,
			mlTensor_t y, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		dim3 blockDim(32, 8);
		dim3 gridDim_x1((last_dim + 31) / 32, std::min(256, (first_dim + 7) / 8));
		dim3 gridDim_x4((last_dim + 127) / 128, std::min(256, (first_dim + 7) / 8));

		cudaStream_t stream = ml::cuda::Context::getStream(context);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_inference<half, 4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(w), data<half>(stats), first_dim, last_dim, act);
				else
					kernel_batchnorm_inference<half, 1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(w), data<half>(stats), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_inference<float, 4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(w), data<float>(stats), first_dim, last_dim, act);
				else
					kernel_batchnorm_inference<float, 1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(w), data<float>(stats), first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, float beta, mlTensor_t y,
			mlTensor_t running_stats, mlActivationType_t act)
	{
		assert(w.rank == 2);
		assert(running_stats.rank == 1);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		assert(w.dim[0] == 2 && w.dim[1] == last_dim);
		assert(running_stats.dim[0] == 3 * last_dim);

		AvgVarStats<float> *workspace = ml::cuda::Context::getWorkspace<AvgVarStats<float>>(context);
		const int workspace_first_dim = std::min((size_t) 256,
				ml::cuda::Context::getWorkspaceSize(context) / (sizeof(AvgVarStats<float> ) * last_dim));
		assert(workspace_first_dim > 0);

		AvgVarStats<float> *running_stats_ptr = data<AvgVarStats<float>>(running_stats);

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, workspace_first_dim);
		dim3 gridDim1_x4((last_dim + 127) / 128, workspace_first_dim);

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_forward_avg_var_1_v2<half, 4> <<<gridDim1_x4, blockDim1, 0, stream >>>(workspace, data<half>(x), first_dim,
							last_dim);
				else
					kernel_batchnorm_forward_avg_var_1_v2<half, 1> <<<gridDim1_x1, blockDim1, 0, stream >>>(workspace, data<half>(x), first_dim,
							last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_forward_avg_var_1_v2<float, 4> <<<gridDim1_x4, blockDim1, 0, stream >>>(workspace, data<float>(x), first_dim,
							last_dim);
				else
					kernel_batchnorm_forward_avg_var_1_v2<float, 1> <<<gridDim1_x1, blockDim1, 0, stream >>>(workspace, data<float>(x), first_dim,
							last_dim);
				break;
			}
			default:
				break;
		}
		cudaDeviceSynchronize();
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_batchnorm_forward_avg_var_2<<<gridDim2, blockDim2, 0, stream>>>(running_stats_ptr, workspace, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		calculate_avg_var(stream, running_stats_ptr, workspace, x.dtype, last_dim);

		const mlTensor_t stats = make_tensor(workspace, w.dtype, { 2, last_dim });
		cuda_batchnorm_inference(context, alpha, x, w, stats, beta, y, act);
	}
	void cuda_batchnorm_backward(mlContext_t context, float alpha, const mlTensor_t x, mlTensor_t dy, const mlTensor_t w,
			const mlTensor_t running_stats, float beta_dx, mlTensor_t dx, float beta_dw, mlTensor_t dw, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		float *avg_var = ml::cuda::Context::getWorkspace<float>(context);
		float *d_sigma_mu = avg_var + 2 * last_dim;
		const int workspace_first_dim = std::min((size_t) 256, ml::cuda::Context::getWorkspaceSize(context) / (2 * sizeof(float) * last_dim) - 2);

		cudaStream_t stream = ml::cuda::Context::getStream(context);
		const AvgVarStats<float> *running_stats_ptr = data<AvgVarStats<float>>(running_stats);

		calculate_avg_var(stream, running_stats_ptr, avg_var, x.dtype, last_dim);

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, workspace_first_dim);
		dim3 gridDim1_x4((last_dim + 127) / 128, workspace_first_dim);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_delta_1_v2<half, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dy),
							data<half>(w), getPointer<half>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_backward_delta_1_v2<half, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dy),
							data<half>(w), getPointer<half>(avg_var), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_delta_1_v2<float, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(d_sigma_mu, data<float>(x),
							data<float>(dy), data<float>(w), getPointer<float>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_backward_delta_1_v2<float, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(d_sigma_mu, data<float>(x),
							data<float>(dy), data<float>(w), getPointer<float>(avg_var), first_dim, last_dim, act);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_batchnorm_backward_delta_2_v2<<<gridDim2, blockDim2, 0, stream>>>(d_sigma_mu, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim3(32, 8);
		dim3 gridDim3_x1((last_dim + 31) / 32, std::min(256, (first_dim + 7) / 8));
		dim3 gridDim3_x4((last_dim + 127) / 128, std::min(256, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_v2<half, 4> <<<gridDim3_x4, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
							data<half>(dy), data<half>(w), getPointer<half>(avg_var), data<half>(dw), first_dim, last_dim, alpha, beta_dx, beta_dw);
				else
					kernel_batchnorm_backward_v2<half, 1> <<<gridDim3_x1, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
							data<half>(dy), data<half>(w), getPointer<half>(avg_var), data<half>(dw), first_dim, last_dim, alpha, beta_dx, beta_dw);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_v2<float, 4> <<<gridDim3_x4, blockDim3, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dx),
							data<float>(dy), data<float>(w), getPointer<float>(avg_var), data<float>(dw), first_dim, last_dim, alpha, beta_dx,
							beta_dw);
				else
					kernel_batchnorm_backward_v2<float, 1> <<<gridDim3_x1, blockDim3, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dx),
							data<float>(dy), data<float>(w), getPointer<float>(avg_var), data<float>(dw), first_dim, last_dim, alpha, beta_dx,
							beta_dw);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_batchnorm_inference_old(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, float beta, mlTensor_t y,
			mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		dim3 blockDim(32, 8);
		dim3 gridDim((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));

		cudaStream_t stream = ml::cuda::Context::getStream(context);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_inference<<<gridDim, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), data<half>(w), first_dim,
						last_dim, act);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_inference<<<gridDim, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x), data<float>(w), first_dim,
						last_dim, act);
				break;
			default:
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_forward_old(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, float beta, mlTensor_t y,
			mlTensor_t running_stats, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		AvgVarStats<float> *workspace = ml::cuda::Context::getWorkspace<AvgVarStats<float>>(context);
		const int workspace_first_dim = std::min((size_t) 256,
				ml::cuda::Context::getWorkspaceSize(context) / (sizeof(AvgVarStats<float> ) * last_dim));
		assert(workspace_first_dim > 0);

		AvgVarStats<float> *running_stats_ptr = data<AvgVarStats<float>>(running_stats);

		dim3 blockDim(32, 32);
		dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_forward_avg_var_1<<<gridDim1, blockDim, 0,stream >>>(workspace, data<half>(x), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_forward_avg_var_1<<<gridDim1, blockDim, 0,stream >>>(workspace, data<float>(x), first_dim, last_dim);
				break;
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 gridDim2(gridDim1.x);
		kernel_batchnorm_forward_avg_var_2<<<gridDim2, blockDim, 0, stream>>>(running_stats_ptr, workspace, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim3(32, 8);
		dim3 gridDim3((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_forward<<<gridDim3, blockDim3, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), data<half>(w),
						running_stats_ptr, first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_forward<<<gridDim3, blockDim3, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x), data<float>(w),
						running_stats_ptr, first_dim, last_dim, act);
				break;
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_backward_old(mlContext_t context, float alpha, const mlTensor_t x, mlTensor_t dy, const mlTensor_t w,
			const mlTensor_t running_stats, float beta_dx, mlTensor_t dx, float beta_dw, mlTensor_t dw, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		float *workspace = ml::cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) 1024, ml::cuda::Context::getWorkspaceSize(context) / (2 * sizeof(float) * last_dim));

		const AvgVarStats<float> *running_stats_ptr = data<AvgVarStats<float>>(running_stats);

		dim3 blockDim(32, 8);
		dim3 gridDim1((last_dim + blockDim.x - 1) / blockDim.x, workspace_first_dim);

		int2 shape1 { first_dim, last_dim };

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_backward_delta_1<<<gridDim1, blockDim, 0, stream>>>(workspace, data<half>(x), data<half>(dy), running_stats_ptr,
						data<half>(w), shape1, act);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_backward_delta_1<<<gridDim1, blockDim, 0, stream>>>(workspace, data<float>(x), data<float>(dy), running_stats_ptr,
						data<float>(w), shape1, act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 gridDim2(gridDim1.x);
		int2 shape2 { workspace_first_dim, last_dim };
		kernel_batchnorm_backward_delta_2<<<gridDim2, dim3(32, 32), 0, stream>>>(workspace, shape2);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 gridDim3((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_backward<<<gridDim3, blockDim, 0, stream>>>(workspace, data<half>(x), data<half>(dx), data<half>(dy), data<half>(w),
						data<half>(dw), running_stats_ptr, shape1, alpha, beta_dx, beta_dw);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_backward<<<gridDim3, blockDim, 0, stream>>>(workspace, data<float>(x), data<float>(dx), data<float>(dy),
						data<float>(w), data<float>(dw), running_stats_ptr, shape1, alpha, beta_dx, beta_dw);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_update(mlContext_t context, const mlTensor_t running_stat, mlTensor_t weights, bool use_gamma, bool use_beta)
	{
		const int first_dim = get_first_dim(running_stat);
		const int last_dim = get_last_dim(running_stat) / 3;

		dim3 blockDim(32, 8);
		dim3 gridDim((last_dim + 31) / 32);
		switch (weights.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_batchnorm_update <<<gridDim, blockDim, 0,ml::cuda::Context::getStream(context)>>>(data<AvgVarStats<float>>(running_stat),
						data<half>(weights), first_dim, last_dim, use_gamma, use_beta);
				break;
			case DTYPE_FLOAT32:
				kernel_batchnorm_update <<<gridDim, blockDim, 0,ml::cuda::Context::getStream(context)>>>(data<AvgVarStats<float>>(running_stat),
						data<float>(weights), first_dim, last_dim, use_gamma, use_beta);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights)
	{
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		if (shape.rank == 3)
		{ // depthwise conv2D
			const int first_dim = volume_without_last_dim(shape);
			const int last_dim = get_last_dim(shape);
			dim3 blockDim(256);
			dim3 gridDim(first_dim);

			kernel_fold_batchnorm_3D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, getPointer<float>(layer_weights),
					getPointer<float>(layer_bias), getPointer<float>(batchnorm_weights));
			assert(cudaGetLastError() == cudaSuccess);
		}
		else
		{
			const int first_dim = get_first_dim(shape);
			const int last_dim = volume_without_first_dim(shape);
			dim3 blockDim(256);
			dim3 gridDim(first_dim);

			kernel_fold_batchnorm_4D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, getPointer<float>(layer_weights),
					getPointer<float>(layer_bias), getPointer<float>(batchnorm_weights));
			assert(cudaGetLastError() == cudaSuccess);
		}
	}

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

//		float *workspace =ml::cuda::Context::getWorkspace<float>(context);
//		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 2048),
//				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));

//		dim3 blockDim(128);
//		dim3 gridDim(workspace_first_dim);
//
//		cudaStream_t stream =ml::cuda::Context::getStream(context);
//
//		float *partial_weights_update = workspace;
//		float *partial_bias_update = workspace + workspace_first_dim * last_dim;
//
//		kernel_layernorm_backward<1><<<gridDim, blockDim, 0, stream >>>(getPointer<float>(input), getPointer<float>(gradient_prev),
//				getPointer<float>(gradient_next), getPointer<float>(weights), partial_weights_update, partial_bias_update, first_dim, last_dim);
//		assert(cudaGetLastError() == cudaSuccess);

		cudaStream_t stream = ml::cuda::Context::getStream(context);

//		dim3 blockDim(32, 8);
//
//		float *workspace =ml::cuda::Context::getWorkspace<float>(context);
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

	void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(256);
		dim3 gridDim(std::min(1024, first_dim));

		cudaStream_t stream = ml::cuda::Context::getStream(context);

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

		float *workspace = ml::cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min((first_dim + (int) blockDim.y - 1) / (int) blockDim.y, 128),
				cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 gridDim(1, workspace_first_dim);

		cudaStream_t stream = ml::cuda::Context::getStream(context);

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

