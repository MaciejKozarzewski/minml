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

#include <cmath>
#include <algorithm>
#include <cassert>

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
	class AvgVarStats
	{
			T samples = static_cast<T>(0);
			T M = static_cast<T>(0); // mean
			T M2 = static_cast<T>(0); // variance
		public:
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

			__device__ void merge_with(const AvgVarStats<T> &rhs) noexcept
			{
				assert(this->samples >= static_cast<T>(0) && rhs.samples >= static_cast<T>(0));
				if (rhs.samples == static_cast<T>(0))
					return;
				if (this->samples == static_cast<T>(0))
				{
					this->samples = rhs.samples;
					this->M = rhs.M;
					this->M2 = rhs.M2;
				}
				else
				{
					const T total_samples = this->samples + rhs.samples;
					const T total_M = (this->samples * this->M + rhs.samples * rhs.M) / total_samples;
					const T total_M2 = this->M2 + rhs.M2 + square(this->M - rhs.M) * (this->samples * rhs.samples) / total_samples;
					this->samples = total_samples;
					this->M = total_M;
					this->M2 = total_M2;
				}
			}
	};

	using namespace ml;
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
	__global__ void kernel_batchnorm_forward_avg_var_1(AvgVarStats<float> *__restrict__ workspace, const float *__restrict__ input, int first_dim,
			int last_dim)
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
		if (threadIdx.y == 0 and tid < last_dim)
			workspace[blockIdx.y * last_dim + tid] = shared_stats[threadIdx.x];
	}
	__global__ void kernel_batchnorm_forward_avg_var_2(float *__restrict__ running_stat, const AvgVarStats<float> *__restrict__ workspace,
			int first_dim, int last_dim)
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
		if (threadIdx.y == 0 and tid < last_dim)
		{
			running_stat[tid] = shared_stats[threadIdx.x].get_average();
			running_stat[last_dim + tid] = shared_stats[threadIdx.x].get_variance();
		}
	}

	__global__ void kernel_batchnorm_forward(const float *weights, const float *input, float *output, const float *running_stats, int2 shape,
			mlActivationType_t act)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < shape.y)
		{
			/* weights rows are:
			 * mean
			 * variance
			 * gamma
			 * beta
			 */
			const float mean = get_mean(running_stats, tid, shape.y);
			const float stddev = get_stddev(running_stats, tid, shape.y);
			const float gamma = get_gamma(weights, tid, shape.y);
			const float beta = get_beta(weights, tid, shape.y);

			const float scale = gamma / stddev;
			const float shift = -mean * scale + beta;

			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < shape.x; i += gridDim.y * blockDim.y)
			{
				float tmp = input[i * shape.y + tid] * scale + shift;
				if (act == ACTIVATION_RELU)
					tmp = max(0.0f, tmp);
				if (act == ACTIVATION_TANH)
					tmp = tanh(tmp);
				if (act == ACTIVATION_SIGMOID)
					tmp = 1.0f / (1.0f + exp(-tmp));
				output[i * shape.y + tid] = tmp;
			}
		}
	}

	__global__ void kernel_batchnorm_inference(const float *weights, const float *input, float *output, int2 shape, mlActivationType_t act)
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
				if (act == ACTIVATION_RELU)
					tmp = max(0.0f, tmp);
				if (act == ACTIVATION_TANH)
					tmp = tanh(tmp);
				if (act == ACTIVATION_SIGMOID)
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
			const float *running_stats, int2 shape, mlActivationType_t act)
	{
		__shared__ float d_sigma[32 * 32];
		__shared__ float d_mu[32 * 32];
		const int tid = blockIdx.x * 32 + threadIdx.x;

		float d_sigma_acc = 0.0f, d_mu_acc = 0.0f;
		if (tid < shape.y)
		{
			const float mean = get_mean(running_stats, tid, shape.y);
			const float stddev = get_stddev(running_stats, tid, shape.y);
			for (int i = 32 * blockIdx.y + threadIdx.y; i < shape.x; i += 32 * gridDim.y)
			{
				const int tmp_idx = i * shape.y + tid;
				if (act == ACTIVATION_RELU and output[tmp_idx] == 0.0f)
					gradient_next[tmp_idx] = 0.0f;
				if (act == ACTIVATION_TANH)
					gradient_next[tmp_idx] *= (1.0f - output[tmp_idx]) * (1.0f + output[tmp_idx]);
				if (act == ACTIVATION_SIGMOID)
					gradient_next[tmp_idx] *= output[tmp_idx] * (1.0f - output[tmp_idx]);
				d_sigma_acc += gradient_next[tmp_idx] * (input[tmp_idx] - mean) / stddev;
				d_mu_acc += gradient_next[tmp_idx];
			}
		}
		d_sigma[threadIdx.y * 32 + threadIdx.x] = d_sigma_acc;
		d_mu[threadIdx.y * 32 + threadIdx.x] = d_mu_acc;

		__syncthreads();
		reduce_add_32x32_dual(d_sigma, d_mu);
		if (threadIdx.y == 0 and tid < shape.y)
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
		if (threadIdx.y == 0 and tid < shape.y)
		{
			workspace[tid] = storage_d_sigma[threadIdx.x];
			workspace[shape.y + tid] = storage_d_mu[threadIdx.x];
		}
	}

	__global__ void kernel_batchnorm_backward_1(const float *workspace, const float *input, float *gradient_prev, const float *gradient_next,
			const float *weights, float *weight_update, const float *running_stats, int2 shape)
	{
		// avg, stddev, gamma, d_sigma, d_mu
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < shape.y)
		{
			const float mean = get_mean(running_stats, tid, shape.y);
			const float stddev = get_stddev(running_stats, tid, shape.y);
			const float gamma = get_gamma(weights, tid, shape.y);

			float d_sigma = workspace[tid];
			float d_mu = workspace[shape.y + tid];
			if (blockIdx.y == 0 and threadIdx.y == 0)
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

	__global__ void kernel_batchnorm_update(const float *running_stat, float *weights, int first_dim, int last_dim, bool use_gamma, bool use_beta)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < last_dim)
		{
			float mean_avg = 0.0f, mean_var = 0.0f;
			for (int i = 0; i < first_dim; i++)
			{
				mean_avg += running_stat[i * 2 * last_dim + tid];
				mean_var += running_stat[(i * 2 + 1) * last_dim + tid];
			}
			weights[0 * last_dim + tid] = mean_avg / first_dim; // running mean average
			weights[1 * last_dim + tid] = mean_var / first_dim; // running mean variance
			if (not use_gamma)
				weights[2 * last_dim + tid] = 1.0f; // gamma
			if (not use_beta)
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

		float *running_stats_ptr = getPointer<float>(running_stats) + running_stat_idx * 2 * last_dim;

		dim3 blockDim(32, 32);
		dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);

		int2 shape1 { first_dim, last_dim };
		dim3 gridDim2(gridDim1.x);
		int2 shape2 { workspace_first_dim, last_dim };
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

		const float *running_stats_ptr = getPointer<float>(running_stats) + running_stat_idx * 2 * last_dim;

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
		const int last_dim = get_last_dim(shape) / 2;
		int2 dim { first_dim, last_dim };

		dim3 blockDim(256);
		dim3 gridDim(std::max(1u, (last_dim + blockDim.x - 1) / blockDim.x));
		kernel_batchnorm_update<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<float>(running_stat),
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

} /* namespace ml */

