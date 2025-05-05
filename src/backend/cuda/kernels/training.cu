/*
 * training.cu
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>

namespace
{
	using namespace vectors;

	__device__ float round_small_to_zero(float x)
	{
		const uint32_t exponent = reinterpret_cast<uint32_t*>(&x)[0] & 0x7f800000;
		return (exponent == 0) ? 0.0f : x;
	}
	__device__ float sign(float x)
	{
		return (x >= 0.0f) ? 1.0f : -1.0f;
	}
	__device__ float safe_log(float x)
	{
		return logf(1.0e-8f + x);
	}
	__device__ float cross_entropy(float output, float target)
	{
		return -target * safe_log(output) - (1.0f - target) * safe_log(1.0f - output);
	}
	__device__ float square(float x)
	{
		return x * x;
	}
	__device__ float bounded_pow(float x, float y, float min)
	{
		assert(0 < x && x < 1);
		assert(min > 0);
		const float max_y = std::log(min) / std::log(x);
		return (y >= max_y) ? 0.0f : std::pow(x, y);
	}

	template<typename T>
	__global__ void kernel_loss_gradient(float beta, T *gradient, float alpha, const T *output, const T *target, const T *mask, int elements)
	{
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;
		for (int i = tid; i < elements; i += stride)
		{
			const float m = (mask == nullptr) ? 1.0f : static_cast<float>(mask[i]);
			const float o = static_cast<float>(output[i]);
			const float t = static_cast<float>(target[i]);
			float grad = m * alpha * (o - t);
			if (beta != 0.0f)
				grad += beta * static_cast<float>(gradient[i]);
			gradient[i] = grad;
		}
	}
	template<typename T>
	__global__ void kernel_CE_loss_step_1(float *workspace, const T *output, const T *target, const T *mask, int elements)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		float acc = 0.0f;
		for (int i = tid; i < elements; i += stride)
		{
			const float m = (mask == nullptr) ? 1.0f : static_cast<float>(mask[i]);
			const float o = static_cast<float>(output[i]);
			const float t = static_cast<float>(target[i]);
			acc += m * max(0.0f, cross_entropy(o, t) - cross_entropy(t, t));
		}
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = sum;
	}
	template<typename T>
	__global__ void kernel_MSE_loss_step_1(float *workspace, const T *output, const T *target, const T *mask, int elements)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			const float m = (mask == nullptr) ? 1.0f : static_cast<float>(mask[i]);
			const float o = static_cast<float>(output[i]);
			const float t = static_cast<float>(target[i]);
			acc += m * square(o - t);
		}
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = sum;
	}
	__global__ void kernel_loss_step_2(float *workspace, int elements)
	{
		assert(gridDim.x == 1);
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			acc += workspace[i];
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[0] = sum;
	}

	struct radam_tensors
	{
			const float *gradient;
			float *weight;
			float *momentum;
			float *variance;
			half *weight_copy;
			int64_t elements;
	};
	__global__ void kernel_fused_learn_radam(float scale, radam_tensors *tensors, float learning_rate, float beta1, float beta2, int step,
			float weight_decay)
	{
		const float pow_beta1 = bounded_pow(beta1, step, 1.0e-8f);
		const float pow_beta2 = bounded_pow(beta2, step, 1.0e-8f);
		const float p_inf = 2.0f / (1.0f - beta2) - 1.0f;
		const float p = p_inf - 2.0f * step * pow_beta2 / (1.0f - pow_beta2);
		float r = 1.0f;
		if (p > 4.0f)
			r = sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));

		const float *gradient = tensors[blockIdx.y].gradient;
		float *weight = tensors[blockIdx.y].weight;
		float *momentum = tensors[blockIdx.y].momentum;
		float *variance = tensors[blockIdx.y].variance;
		const int elements = tensors[blockIdx.y].elements;
		half *weight_copy = tensors[blockIdx.y].weight_copy;

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			float w = weight[i];
			float g = scale * gradient[i] + weight_decay * w;
			float m = momentum[i];
			float v = variance[i];

			m = beta1 * m + (1.0f - beta1) * g;
			v = beta2 * v + (1.0f - beta2) * square(g);

			float correction = 1.0f;
			if (p > 4.0f)
				correction = sqrt((1.0f - pow_beta2) / (v + 1.0e-8f)) * r;

			w = round_small_to_zero(w - learning_rate * correction * m / (1.0f - pow_beta1));

			momentum[i] = m;
			variance[i] = v;
			weight[i] = w;
			if (weight_copy != nullptr)
				weight_copy[i] = w;
		}
	}
	__global__ void kernel_fused_learn_lion(float scale, radam_tensors *tensors, float learning_rate, float beta1, float beta2, int step,
			float weight_decay)
	{
		const float *gradient = tensors[blockIdx.y].gradient;
		float *weight = tensors[blockIdx.y].weight;
		float *momentum = tensors[blockIdx.y].momentum;
		const int elements = tensors[blockIdx.y].elements;
		half *weight_copy = tensors[blockIdx.y].weight_copy;

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			float w = weight[i];
			const float g = scale * gradient[i] + weight_decay * w;
			const float m = momentum[i];

			const float c = beta1 * m + (1.0f - beta1) * g;

			w = round_small_to_zero(w - learning_rate * sign(c));

			momentum[i] = beta2 * m + (1.0f - beta2) * g;
			weight[i] = w;
			if (weight_copy != nullptr)
				weight_copy[i] = w;
		}
	}

	struct nan_inf_tensors
	{
			const void *data;
			int32_t elements;
			uint32_t flag;
	};

	__device__ uint32_t is_nan_or_inf(float x)
	{
		return static_cast<uint32_t>(isnan(x)) + (static_cast<uint32_t>(isinf(x)) << 1u);
	}
	__device__ uint32_t is_nan_or_inf(half x)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		return static_cast<uint32_t>(__hisnan(x)) + (static_cast<uint32_t>(__hisinf(x) & 1) << 1u);
#else
		return 0u;
#endif
	}
	template<typename T, int N>
	__device__ uint32_t check_nan_inf(const T *ptr, int elements)
	{
		assert(elements % N == 0);
		uint32_t result = 0;
		for (int i = N * threadIdx.x; i < elements; i += N * blockDim.x)
		{
			const vec<T, N> tmp(ptr + i);
			for (int n = 0; n < N; n++)
				result |= is_nan_or_inf(tmp[n]);
		}
		return result;
	}
	template<typename T>
	__global__ void kernel_fused_is_nan_or_inf(nan_inf_tensors *tensors)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		const T *data = reinterpret_cast<const T*>(tensors[blockIdx.x].data);
		const int elements = tensors[blockIdx.x].elements;

		uint32_t local_flag = 0;
		if (elements % 4 == 0)
			local_flag = check_nan_inf<T, 4>(data, elements);
		else
			local_flag = check_nan_inf<T, 1>(data, elements);

		const uint32_t flag = cg::reduce(tile, local_flag, cg::bit_or<uint32_t>());
		if (threadIdx.x == 0)
			tensors[blockIdx.x].flag = flag;
	}

	float reduce_loss_step_2(cudaStream_t stream, float *workspace, int elements)
	{
		kernel_loss_step_2<<<1, 256, 0, stream>>>(workspace, elements);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result;
	}
}

namespace ml
{
	float cuda_mean_squared_loss(mlContext_t context, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask)
	{
		const int elements = volume(output);

		assert(ml::cuda_backend::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));

		float *workspace = ml::cuda_backend::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (output.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_MSE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, data<half>(output), data<half>(target), data<half>(mask),
						elements);
				break;
			case DTYPE_FLOAT32:
				kernel_MSE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, data<float>(output), data<float>(target), data<float>(mask),
						elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		return reduce_loss_step_2(stream, workspace, gridDim.x);
	}
	float cuda_cross_entropy_loss(mlContext_t context, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask)
	{
		const int elements = volume(output);

		assert(ml::cuda_backend::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));

		float *workspace = ml::cuda_backend::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (output.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_CE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, data<half>(output), data<half>(target), data<half>(mask),
						elements);
				break;
			case DTYPE_FLOAT32:
				kernel_CE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, data<float>(output), data<float>(target), data<float>(mask),
						elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		return reduce_loss_step_2(stream, workspace, gridDim.x);
	}
	void cuda_mean_squared_gradient(mlContext_t context, float alpha, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask,
			float beta, mlTensor_t gradient)
	{
		const int elements = volume(output);

		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (output.dtype)
		{
			case DTYPE_FLOAT16:
			{
				kernel_loss_gradient <<<gridDim, blockDim, 0, stream>>>(beta, data<half>(gradient), alpha, data<half>(output), data<half>(target),
						data<half>(mask), elements);
				break;
			}
			case DTYPE_FLOAT32:
			{
				kernel_loss_gradient <<<gridDim, blockDim, 0, stream>>>(beta, data<float>(gradient), alpha, data<float>(output), data<float>(target),
						data<float>(mask), elements);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_cross_entropy_gradient(mlContext_t context, float alpha, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask,
			float beta, mlTensor_t gradient)
	{
		cuda_mean_squared_gradient(context, alpha, output, target, mask, beta, gradient);
	}

	void cuda_fused_radam_optimize(mlContext_t context, float scale, const mlTensor_t *gradients, mlTensor_t *weights, mlTensor_t *momentums,
			mlTensor_t *variances, mlTensor_t *weights_copy, float learning_rate, float beta1, float beta2, int step, int num_tensors,
			float weight_decay)
	{
		assert(step > 0);
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		radam_tensors *cpu_workspace = reinterpret_cast<radam_tensors*>(ml::cuda_backend::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			assert(is_fp32(gradients[i]));
			assert(is_fp32(weights[i]));
			assert(is_fp32(momentums[i]));
			assert(is_fp32(variances[i]));
			cpu_workspace[i].gradient = data<float>(gradients[i]);
			cpu_workspace[i].weight = data<float>(weights[i]);
			cpu_workspace[i].momentum = data<float>(momentums[i]);
			cpu_workspace[i].variance = data<float>(variances[i]);
			if (weights_copy != nullptr)
			{
				assert(is_fp16(weights_copy[i]));
				cpu_workspace[i].weight_copy = data<half>(weights_copy[i]);
			}
			else
				cpu_workspace[i].weight_copy = nullptr;
			cpu_workspace[i].elements = volume(gradients[i]);
		}

		radam_tensors *device_workspace = ml::cuda_backend::Context::getWorkspace<radam_tensors>(context);
		cudaError_t status = cudaMemcpyAsync(device_workspace, cpu_workspace, sizeof(radam_tensors) * num_tensors, cudaMemcpyHostToDevice, stream);
		assert(status == cudaSuccess);

		dim3 blockDim(256);
		dim3 gridDim(32, num_tensors);
		kernel_fused_learn_radam<<<gridDim, blockDim, 0, stream>>>(scale, device_workspace, learning_rate, beta1, beta2, step, weight_decay);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_fused_lion_optimize(mlContext_t context, float scale, const mlTensor_t *gradients, mlTensor_t *weights, mlTensor_t *momentums,
			mlTensor_t *weights_copy, float learning_rate, float beta1, float beta2, int step, int num_tensors, float weight_decay)
	{
		assert(step > 0);
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		radam_tensors *cpu_workspace = reinterpret_cast<radam_tensors*>(ml::cuda_backend::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			assert(is_fp32(gradients[i]));
			assert(is_fp32(weights[i]));
			assert(is_fp32(momentums[i]));
			cpu_workspace[i].gradient = data<float>(gradients[i]);
			cpu_workspace[i].weight = data<float>(weights[i]);
			cpu_workspace[i].momentum = data<float>(momentums[i]);
			cpu_workspace[i].variance = nullptr;
			if (weights_copy != nullptr)
			{
				assert(is_fp16(weights_copy[i]));
				cpu_workspace[i].weight_copy = data<half>(weights_copy[i]);
			}
			else
				cpu_workspace[i].weight_copy = nullptr;
			cpu_workspace[i].elements = volume(gradients[i]);
		}

		radam_tensors *device_workspace = ml::cuda_backend::Context::getWorkspace<radam_tensors>(context);
		cudaError_t status = cudaMemcpyAsync(device_workspace, cpu_workspace, sizeof(radam_tensors) * num_tensors, cudaMemcpyHostToDevice, stream);
		assert(status == cudaSuccess);

		dim3 blockDim(256);
		dim3 gridDim(32, num_tensors);
		kernel_fused_learn_lion<<<gridDim, blockDim, 0, stream>>>(scale, device_workspace, learning_rate, beta1, beta2, step, weight_decay);
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_fused_is_nan_or_inf(mlContext_t context, const mlTensor_t *tensor, int *result, int num_tensors)
	{
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		nan_inf_tensors *cpu_workspace = reinterpret_cast<nan_inf_tensors*>(ml::cuda_backend::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			cpu_workspace[i].data = tensor[i].data;
			cpu_workspace[i].elements = volume(tensor[i]);
			cpu_workspace[i].flag = 0u;
		}

		nan_inf_tensors *device_workspace = ml::cuda_backend::Context::getWorkspace<nan_inf_tensors>(context);
		cudaError_t status = cudaMemcpyAsync(device_workspace, cpu_workspace, sizeof(nan_inf_tensors) * num_tensors, cudaMemcpyHostToDevice, stream);
		assert(status == cudaSuccess);

		dim3 blockDim(256);
		dim3 gridDim(num_tensors);
		switch (tensor[0].dtype)
		{
			case DTYPE_FLOAT16:
				kernel_fused_is_nan_or_inf<half> <<<gridDim, blockDim, 0, stream>>>(device_workspace);
				break;
			case DTYPE_FLOAT32:
				kernel_fused_is_nan_or_inf<float> <<<gridDim, blockDim, 0, stream>>>(device_workspace);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		status = cudaMemcpyAsync(cpu_workspace, device_workspace, sizeof(nan_inf_tensors) * num_tensors, cudaMemcpyDeviceToHost, stream);
		assert(status == cudaSuccess);
		status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);

		for (int i = 0; i < num_tensors; i++)
			result[i] = cpu_workspace[i].flag;
	}

	void cuda_combined_loss(mlContext_t context, float *result, const mlTensor_t *outputs, const mlTensor_t *targets, const mlTensor_t *masks,
			const float *weights)
	{

	}
} /* namespace ml */

