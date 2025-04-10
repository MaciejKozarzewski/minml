/*
 * batchnorm.cu
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

	template<int N, typename T>
	__global__ void kernel_batchnorm_inference(float beta, T *output, float alpha, const T *input, const T *weights, const T *bias,
			const float *avg_var, int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		if (tid < last_dim)
		{
//			const vec<T, N> _mean = load_vec<T, N>(avg_var + tid);
//			const vec<T, N> _variance = load_vec<T, N>(avg_var + tid + last_dim);
//			const vec<T, N> _gamma = (weights != nullptr) ? vec<T, N>(weights + tid) : one<T, N>();
//			const vec<T, N> _beta = (bias != nullptr) ? vec<T, N>(bias + tid) : zero<T, N>();
//
//			const vec<T, N> epsilon(1.0e-6f);
//
//			const vec<T, N> scale = _gamma / vectors::sqrt(_variance + epsilon);
//			const vec<T, N> shift = -_mean * scale + _beta;
//
//			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
//			{
//				const int tmp_idx = i * last_dim + tid;
//				vec<T, N> tmp(input + tmp_idx);
//				tmp = tmp * scale + shift;
//				switch (act)
//				{
//					case ml::ACTIVATION_SIGMOID:
//						tmp = vectors::sigmoid(tmp);
//						break;
//					case ml::ACTIVATION_TANH:
//						tmp = vectors::tanh(tmp);
//						break;
//					case ml::ACTIVATION_RELU:
//						tmp = vectors::relu(tmp);
//						break;
//					case ml::ACTIVATION_LEAKY_RELU:
//						tmp = select(tmp > zero<T, N>(), tmp, tmp * vec<T, N>(0.1f));
//						break;
//				}
//				tmp *= vec<T, N>(alpha);
//				if (beta != 0.0f)
//					tmp += vec<T, N>(beta) * vec<T, N>(output + tmp_idx);
//				tmp.store(output + tmp_idx);
//			}

			const vec<float, N> _mean = load_vec<float, N>(avg_var + tid);
			const vec<float, N> _variance = load_vec<float, N>(avg_var + tid + last_dim);
			const vec<float, N> _gamma = (weights != nullptr) ? load_vec<float, N>(weights + tid) : one<float, N>();
			const vec<float, N> _beta = (bias != nullptr) ? load_vec<float, N>(bias + tid) : zero<float, N>();

			const vec<float, N> epsilon(1.0e-6f);

			const vec<float, N> scale = _gamma / vectors::sqrt(_variance + epsilon);
			const vec<float, N> shift = -_mean * scale + _beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int tmp_idx = i * last_dim + tid;
				vec<float, N> tmp = load_vec<float, N>(input + tmp_idx);
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
					case ml::ACTIVATION_LEAKY_RELU:
						tmp = select(tmp > zero<float, N>(), tmp, tmp * vec<float, N>(0.1f));
						break;
				}
				tmp *= vec<float, N>(alpha);
				if (beta != 0.0f)
					tmp += vec<float, N>(beta) * load_vec<float, N>(output + tmp_idx);
				store_vec(output + tmp_idx, tmp);
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
	template<int N, typename T>
	__global__ void kernel_batchnorm_forward_avg_var_1(AvgVarStats<float> *workspace, const T *input, int first_dim, int last_dim)
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
				const vec<float, N> tmp = load_vec<float, N>(input + i * last_dim + tid);
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
	__global__ void kernel_batchnorm_forward_avg_var_2(AvgVarStats<float> *running_stats, const AvgVarStats<float> *workspace, int first_dim,
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
	template<int N, typename T>
	__global__ void kernel_batchnorm_backward_delta_1(float *workspace, const T *input, T *gradient_next, const T *weights, const T *bias,
			const float *avg_var, int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(blockDim.x == 32 && blockDim.y == 8);
		__shared__ float d_sigma[8 * 32 * N];
		__shared__ float d_mu[8 * 32 * N];
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);

		vec<float, N> d_sigma_acc(0.0f);
		vec<float, N> d_mu_acc(0.0f);
		if (tid < last_dim)
		{
//			const vec<T, N> epsilon(1.0e-6f);
//			const vec<T, N> _mean = load_vec<T, N>(avg_var + tid);
//			const vec<T, N> _stddev = vectors::sqrt(load_vec<T, N>(avg_var + tid + last_dim) + epsilon);
//			const vec<T, N> _gamma = (weights != nullptr) ? vec<T, N>(weights + tid) : one<T, N>();
//			const vec<T, N> _beta = (bias != nullptr) ? vec<T, N>(bias + tid) : zero<T, N>();
//
//			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//			{
//				const int tmp_idx = i * last_dim + tid;
//				vec<T, N> grad(gradient_next + tmp_idx);
//				const vec<T, N> inp(input + tmp_idx);
//				const vec<T, N> hat_inp = (inp - _mean) / _stddev;
//				vec<T, N> out = _gamma * hat_inp + _beta;
//				switch (act)
//				{
//					case ml::ACTIVATION_SIGMOID:
//						out = vectors::sigmoid(out);
//						grad *= out * (one<T, N>() - out);
//						break;
//					case ml::ACTIVATION_TANH:
//						grad *= (one<T, N>() - square(vectors::tanh(out)));
//						break;
//					case ml::ACTIVATION_RELU:
//						grad = select(out > zero<T, N>(), grad, zero<T, N>());
//						break;
//					case ml::ACTIVATION_LEAKY_RELU:
//						grad = select(out > zero<T, N>(), grad, grad * vec<T, N>(0.1f));
//						break;
//				}
//				d_sigma_acc += convert<float, T, N>(grad * hat_inp);
//				d_mu_acc += convert<float, T, N>(grad);
//				grad.store(gradient_next + tmp_idx);
//			}

			const vec<float, N> epsilon(1.0e-6f);
			const vec<float, N> _mean = load_vec<float, N>(avg_var + tid);
			const vec<float, N> _stddev = vectors::sqrt(load_vec<float, N>(avg_var + tid + last_dim) + epsilon);
			const vec<float, N> _gamma = (weights != nullptr) ? load_vec<float, N>(weights + tid) : one<float, N>();
			const vec<float, N> _beta = (bias != nullptr) ? load_vec<float, N>(bias + tid) : zero<float, N>();

			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = i * last_dim + tid;
				vec<float, N> grad = load_vec<float, N>(gradient_next + tmp_idx);
				const vec<float, N> inp = load_vec<float, N>(input + tmp_idx);
				const vec<float, N> hat_inp = (inp - _mean) / _stddev;
				vec<float, N> out = _gamma * hat_inp + _beta;
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						out = vectors::sigmoid(out);
						grad *= out * (one<float, N>() - out);
						break;
					case ml::ACTIVATION_TANH:
						grad *= (one<float, N>() - square(vectors::tanh(out)));
						break;
					case ml::ACTIVATION_RELU:
						grad = select(out > zero<float, N>(), grad, zero<float, N>());
						break;
					case ml::ACTIVATION_LEAKY_RELU:
						grad = select(out > zero<float, N>(), grad, grad * vec<float, N>(0.1f));
						break;
				}
				d_sigma_acc += grad * hat_inp;
				d_mu_acc += grad;
				store_vec(gradient_next + tmp_idx, grad);
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
	__global__ void kernel_batchnorm_backward_delta_2(float *workspace, int first_dim, int last_dim)
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
	}
	template<int N, typename T, typename U>
	__global__ void kernel_batchnorm_backward(const float *d_sigma_mu, const T *input, T *gradient_prev, const T *gradient_next, const T *weights,
			const float *avg_var, U *weight_update, U *bias_update, int first_dim, int last_dim, float alpha, float beta_dx, float beta_dw)
	{
		// avg, stddev, gamma, d_sigma, d_mu
		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		if (tid < last_dim)
		{
			const vec<float, N> epsilon(1.0e-6f);
			const vec<float, N> _mean(avg_var + tid);
			const vec<float, N> _stddev = vectors::sqrt(vec<float, N>(avg_var + tid + last_dim) + epsilon);
			const vec<float, N> _gamma = (weights != nullptr) ? load_vec<float, N>(weights + tid) : one<float, N>();

			vec<float, N> d_sigma = load_vec<float, N>(d_sigma_mu + tid);
			vec<float, N> d_mu = load_vec<float, N>(d_sigma_mu + tid + last_dim);
			if (blockIdx.y == 0 && threadIdx.y == 0)
			{ // only single line can update this
				vec<float, N> ds = d_sigma * vec<float, N>(alpha);
				vec<float, N> dm = d_mu * vec<float, N>(alpha);
				if (beta_dw != 0.0f)
				{
					if (weight_update != nullptr)
						ds += beta_dw * load_vec<float, N>(weight_update + tid);
					if (bias_update != nullptr)
						dm += beta_dw * load_vec<float, N>(bias_update + tid);
				}
				if (weight_update != nullptr)
					store_vec(weight_update + tid, ds);
				if (bias_update != nullptr)
					store_vec(bias_update + tid, dm);
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

	__global__ void kernel_batchnorm_update(const AvgVarStats<float> *running_stat, float *avg_var, int first_dim, int last_dim)
	{
		__shared__ AvgVarStats<float> shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts
		for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < last_dim; j += gridDim.x * blockDim.x)
		{
			const Indexer<3> idx(blockDim.y, blockDim.x, 1);
			AvgVarStats<float> thread_stat;
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
//				printf("%i, %i = %f, %f, %f\n", i, j, running_stat[i * last_dim + j].samples, running_stat[i * last_dim + j].M,
//						running_stat[i * last_dim + j].M2);
				thread_stat.merge_with(running_stat[i * last_dim + j]);
			}

			shared_stats[idx.at(threadIdx.y, threadIdx.x, 0)] = thread_stat;
			__syncthreads();

			combine_stats_vec<1>(shared_stats, idx);
			if (blockIdx.y == 0 && threadIdx.y == 0)
			{
				avg_var[0 * last_dim + j] = shared_stats[idx.at(0, threadIdx.x, 0)].get_average();
				avg_var[1 * last_dim + j] = shared_stats[idx.at(0, threadIdx.x, 0)].get_variance();
			}
		}
	}

	template<typename T>
	__global__ void kernel_fold_batchnorm_4D(int first_dim, int last_dim, T *layer_weights, T *layer_bias, const T *bn_weights, const T *bn_bias,
			const float *bn_avg_var)
	{
		const float mean = bn_avg_var[blockIdx.x];
		const float stddev = sqrt(static_cast<float>(bn_avg_var[blockIdx.x + first_dim]) + 1.0e-6f);
		const float gamma = (bn_weights != nullptr) ? static_cast<float>(bn_weights[blockIdx.x]) : 1.0f;
		const float beta = (bn_bias != nullptr) ? static_cast<float>(bn_bias[blockIdx.x]) : 0.0f;

		const float scale = gamma / stddev;
		const float shift = -mean * scale + beta;
		for (int i = threadIdx.x; i < last_dim; i += blockDim.x)
			layer_weights[blockIdx.x * last_dim + i] *= scale;

		if (threadIdx.x == 0)
			layer_bias[blockIdx.x] = static_cast<float>(layer_bias[blockIdx.x]) * scale + shift;
	}
	template<typename T>
	__global__ void kernel_fold_batchnorm_3D(int first_dim, int last_dim, T *layer_weights, T *layer_bias, const T *bn_weights, const T *bn_bias,
			const float *bn_avg_var)
	{
		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const float stddev = sqrt(static_cast<float>(bn_avg_var[j + last_dim]) + 1.0e-6f);
				const float gamma = (bn_weights != nullptr) ? static_cast<float>(bn_weights[j]) : 1.0f;
				const float scale = gamma / stddev;
				layer_weights[i * last_dim + j] *= scale;

				if (i == 0)
				{
					const float mean = bn_avg_var[j];
					const float beta = (bn_bias != nullptr) ? static_cast<float>(bn_bias[j]) : 0.0f;
					const float shift = -mean * scale + beta;
					layer_bias[j] = static_cast<float>(layer_bias[j]) * scale + shift;
				}
			}
	}

}

namespace ml
{
	void cuda_batchnorm_inference(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b,
			const mlTensor_t avg_var, float beta, mlTensor_t y, mlActivationType_t act)
	{
		assert(avg_var.dtype == DTYPE_FLOAT32);
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
					kernel_batchnorm_inference<4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), data<half>(w),
							data<half>(b), data<float>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_inference<1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), data<half>(w),
							data<half>(b), data<float>(avg_var), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_inference<4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x), data<float>(w),
							data<float>(b), data<float>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_inference<1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x), data<float>(w),
							data<float>(b), data<float>(avg_var), first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y, mlTensor_t running_stats, mlActivationType_t act)
	{
		assert(running_stats.rank == 1);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
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
					kernel_batchnorm_forward_avg_var_1<4> <<<gridDim1_x4, blockDim1, 0, stream >>>(workspace, data<half>(x), first_dim, last_dim);
				else
					kernel_batchnorm_forward_avg_var_1<1> <<<gridDim1_x1, blockDim1, 0, stream >>>(workspace, data<half>(x), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_forward_avg_var_1<4> <<<gridDim1_x4, blockDim1, 0, stream >>>(workspace, data<float>(x), first_dim, last_dim);
				else
					kernel_batchnorm_forward_avg_var_1<1> <<<gridDim1_x1, blockDim1, 0, stream >>>(workspace, data<float>(x), first_dim, last_dim);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_batchnorm_forward_avg_var_2<<<gridDim2, blockDim2, 0, stream>>>(running_stats_ptr, workspace, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_calculate_avg_var<float> <<<1, 1024, 0, stream>>>(ml::getPointer<float>(workspace), running_stats_ptr, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		const mlTensor_t stats = make_tensor(workspace, DTYPE_FLOAT32, { 2, last_dim });
		cuda_batchnorm_inference(context, alpha, x, w, b, stats, beta, y, act);
	}
	void cuda_batchnorm_backward(mlContext_t context, float alpha, const mlTensor_t x, mlTensor_t dy, const mlTensor_t w, const mlTensor_t b,
			const mlTensor_t running_stats, float beta_dx, mlTensor_t dx, float beta_dw, mlTensor_t dw, mlTensor_t db, mlActivationType_t act)
	{
		assert(dw.dtype == db.dtype);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		float *avg_var = ml::cuda::Context::getWorkspace<float>(context);
		float *d_sigma_mu = avg_var + 2 * last_dim;
		const int workspace_first_dim = std::min((size_t) 256, ml::cuda::Context::getWorkspaceSize(context) / (2 * sizeof(float) * last_dim) - 2);

		cudaStream_t stream = ml::cuda::Context::getStream(context);
		const AvgVarStats<float> *running_stats_ptr = data<AvgVarStats<float>>(running_stats);

		kernel_calculate_avg_var<float> <<<1, 1024, 0, stream>>>(avg_var, running_stats_ptr, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, workspace_first_dim);
		dim3 gridDim1_x4((last_dim + 127) / 128, workspace_first_dim);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_delta_1<4> <<<gridDim1_x4, blockDim1, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dy),
							data<half>(w), data<half>(b), getPointer<float>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_backward_delta_1<1> <<<gridDim1_x1, blockDim1, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dy),
							data<half>(w), data<half>(b), getPointer<float>(avg_var), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward_delta_1<4> <<<gridDim1_x4, blockDim1, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dy),
							data<float>(w), data<float>(b), getPointer<float>(avg_var), first_dim, last_dim, act);
				else
					kernel_batchnorm_backward_delta_1<1> <<<gridDim1_x1, blockDim1, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dy),
							data<float>(w), data<float>(b), getPointer<float>(avg_var), first_dim, last_dim, act);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		kernel_batchnorm_backward_delta_2<<<gridDim2, blockDim2, 0, stream>>>(d_sigma_mu, workspace_first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim3(32, 8);
		dim3 gridDim3_x1((last_dim + 31) / 32, std::min(256, (first_dim + 7) / 8));
		dim3 gridDim3_x4((last_dim + 127) / 128, std::min(256, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
				{
					if (is_fp16(dw))
						kernel_batchnorm_backward<4> <<<gridDim3_x4, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
								data<half>(dy), data<half>(w), getPointer<float>(avg_var), data<half>(dw), data<half>(db), first_dim, last_dim, alpha,
								beta_dx, beta_dw );
					else
						kernel_batchnorm_backward<4> <<<gridDim3_x4, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
								data<half>(dy), data<half>(w), getPointer<float>(avg_var), data<float>(dw), data<float>(db), first_dim, last_dim,
								alpha, beta_dx, beta_dw );
				}
				else
				{
					if (is_fp16(dw))
						kernel_batchnorm_backward<1> <<<gridDim3_x1, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
								data<half>(dy), data<half>(w), getPointer<float>(avg_var), data<half>(dw), data<half>(db), first_dim, last_dim, alpha,
								beta_dx, beta_dw );
					else
						kernel_batchnorm_backward<1> <<<gridDim3_x1, blockDim3, 0, stream>>>(d_sigma_mu, data<half>(x), data<half>(dx),
								data<half>(dy), data<half>(w), getPointer<float>(avg_var), data<float>(dw), data<float>(db), first_dim, last_dim,
								alpha, beta_dx, beta_dw );
				}
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_batchnorm_backward<4> <<<gridDim3_x4, blockDim3, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dx), data<float>(dy),
							data<float>(w), getPointer<float>(avg_var), data<float>(dw), data<float>(db), first_dim, last_dim, alpha, beta_dx,
							beta_dw);
				else
					kernel_batchnorm_backward<1> <<<gridDim3_x1, blockDim3, 0, stream>>>(d_sigma_mu, data<float>(x), data<float>(dx), data<float>(dy),
							data<float>(w), getPointer<float>(avg_var), data<float>(dw), data<float>(db), first_dim, last_dim, alpha, beta_dx,
							beta_dw);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_batchnorm_update(mlContext_t context, const mlTensor_t running_stat, mlTensor_t avg_var)
	{
		assert(running_stat.dtype == DTYPE_FLOAT32);
		assert(avg_var.dtype == DTYPE_FLOAT32);
		const int first_dim = get_first_dim(running_stat);
		const int last_dim = get_last_dim(running_stat) / 3;

		cudaStream_t stream = ml::cuda::Context::getStream(context);

		dim3 blockDim(32, 32);
		dim3 gridDim((last_dim + 31) / 32);
		kernel_batchnorm_update <<<gridDim, blockDim, 0, stream>>>(data<AvgVarStats<float>>(running_stat), data<float>(avg_var), first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
		cudaDeviceSynchronize();
	}
	void cuda_fold_batchnorm(mlContext_t context, mlTensor_t layer_weights, mlTensor_t layer_bias, const mlTensor_t bn_weights,
			const mlTensor_t bn_bias, const mlTensor_t bn_avg_var)
	{
		assert(bn_avg_var.dtype == DTYPE_FLOAT32);
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		if (layer_weights.rank == 3)
		{ // depthwise conv2D
			const int first_dim = volume_without_last_dim(layer_weights);
			const int last_dim = get_last_dim(layer_weights);
			dim3 blockDim(256);
			dim3 gridDim(first_dim);

			switch (layer_weights.dtype)
			{
				case DTYPE_FLOAT16:
					kernel_fold_batchnorm_3D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, data<half>(layer_weights), data<half>(layer_bias),
							data<half>(bn_weights), data<half>(bn_bias), data<float>(bn_avg_var));
					break;
				case DTYPE_FLOAT32:
					kernel_fold_batchnorm_3D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, data<float>(layer_weights),
							data<float>(layer_bias), data<float>(bn_weights), data<float>(bn_bias), data<float>(bn_avg_var));
					break;
				default:
					break;
			}
		}
		else
		{
			const int first_dim = get_first_dim(layer_weights);
			const int last_dim = volume_without_first_dim(layer_weights);
			dim3 blockDim(256);
			dim3 gridDim(first_dim);

			switch (layer_weights.dtype)
			{
				case DTYPE_FLOAT16:
					kernel_fold_batchnorm_4D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, data<half>(layer_weights), data<half>(layer_bias),
							data<half>(bn_weights), data<half>(bn_bias), data<float>(bn_avg_var));
					break;
				case DTYPE_FLOAT32:
					kernel_fold_batchnorm_4D<<<gridDim, blockDim, 0, stream>>>(first_dim, last_dim, data<float>(layer_weights),
							data<float>(layer_bias), data<float>(bn_weights), data<float>(bn_bias), data<float>(bn_avg_var));
					break;
				default:
					break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

