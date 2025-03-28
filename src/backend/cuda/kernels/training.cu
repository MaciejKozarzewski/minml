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

	__device__ vec4f round_small_to_zero(vec4f x)
	{
		vec4f result;
		result.x0 = (fabsf(x.x0) < 1.0e-6f) ? 0.0f : x.x0;
		result.x1 = (fabsf(x.x1) < 1.0e-6f) ? 0.0f : x.x1;
		result.x2 = (fabsf(x.x2) < 1.0e-6f) ? 0.0f : x.x2;
		result.x3 = (fabsf(x.x3) < 1.0e-6f) ? 0.0f : x.x3;
		return result;
	}
	__device__ vec1f round_small_to_zero(vec1f x)
	{
		return vec1f((fabsf(x.x0) < 1.0e-6f) ? 0.0f : x.x0);
	}
	__device__ float round_small_to_zero(float x)
	{
		return (fabsf(x) < 1.0e-6f) ? 0.0f : x;
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

	template<int N, typename T>
	__global__ void kernel_learn_radam(float scale, const T *gradient, float *weight, float *momentum, float *variance, int elements,
			float learning_rate, float beta1, float beta2, int step)
	{
		const float pow_beta1 = bounded_pow(beta1, step, 1.0e-8f);
		const float pow_beta2 = bounded_pow(beta2, step, 1.0e-8f);
		const float p_inf = 2.0f / (1.0f - beta2) - 1.0f;
		const float p = p_inf - 2.0f * step * pow_beta2 / (1.0f - pow_beta2);
		float r = 1.0f;
		if (p > 4.0f)
			r = sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));

		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = N * tid; i < elements; i += N * stride)
		{
			vec<float, N> w(weight + i);
			vec<float, N> g = scale * convert<float>(vec<T, N>(gradient + i));
			vec<float, N> m(momentum + i);
			vec<float, N> v(variance + i);

			m = beta1 * m + (1.0f - beta1) * g;
			v = beta2 * v + (1.0f - beta2) * square(g);

			vec<float, N> correction(1.0f);
			if (p > 4.0f)
				correction = sqrt((1.0f - pow_beta2) / (v + 1.0e-8f)) * r;

			w = round_small_to_zero(w - learning_rate * correction * m / (1.0f - pow_beta1));

			m.store(momentum + i);
			v.store(variance + i);
			w.store(weight + i);
		}
	}

	struct radam_tensors
	{
			const void *gradient;
			float *weight;
			float *momentum;
			float *variance;
			int64_t elements;
	};
	template<typename T>
	__global__ void kernel_fused_learn_radam(float scale, radam_tensors *tensors, float learning_rate, float beta1, float beta2, int step)
	{
		const float pow_beta1 = bounded_pow(beta1, step, 1.0e-8f);
		const float pow_beta2 = bounded_pow(beta2, step, 1.0e-8f);
		const float p_inf = 2.0f / (1.0f - beta2) - 1.0f;
		const float p = p_inf - 2.0f * step * pow_beta2 / (1.0f - pow_beta2);
		float r = 1.0f;
		if (p > 4.0f)
			r = sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));

		const T *gradient = reinterpret_cast<const T*>(tensors[blockIdx.y].gradient);
		float *weight = tensors[blockIdx.y].weight;
		float *momentum = tensors[blockIdx.y].momentum;
		float *variance = tensors[blockIdx.y].variance;
		const int elements = tensors[blockIdx.y].elements;

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			float w = weight[i];
			float g = scale * static_cast<float>(gradient[i]);
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
		}
	}

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
	__global__ void kernel_is_nan_or_inf_step_1(uint32_t *workspace, const T *input, int elements)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		const int tid = N * (blockIdx.x * blockDim.x + threadIdx.x);
		const int stride = N * (gridDim.x * blockDim.x);

		uint32_t local_flag = 0;
		for (int i = tid; i < elements; i += stride)
		{
			const vec<T, N> tmp(input + i);
			for (int n = 0; n < N; n++)
				local_flag |= is_nan_or_inf(tmp[n]);
		}
		const uint32_t flag = cg::reduce(tile, local_flag, cg::bit_or<uint32_t>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = flag;
	}
	__global__ void kernel_is_nan_or_inf_step_2(uint32_t *workspace, int elements)
	{
		assert(gridDim.x == 1);
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		uint32_t local_flag = 0;
		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			local_flag |= workspace[i];
		const uint32_t flag = cg::reduce(tile, local_flag, cg::bit_or<uint32_t>());
		if (threadIdx.x == 0)
			workspace[0] = flag;
	}

	template<typename T, int N>
	__global__ void kernel_regularizer_l2(T *gradient, const T *param, float scale, float offset, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = N * tid; i < elements; i += N * stride)
		{
			const vec<T, N> w(param + i);
			vec<T, N> g(gradient + i);
			g += vec<T, N>(scale) * (w - vec<T, N>(offset));
			g.store(gradient + i);
		}
	}

	struct l2_tensors
	{
			void *gradient;
			const void *weight;
			int64_t elements;
	};
	template<typename T>
	__global__ void kernel_fused_regularizer_l2(l2_tensors *tensors, float scale)
	{
		T *gradient = reinterpret_cast<T*>(tensors[blockIdx.y].gradient);
		const T *weight = reinterpret_cast<const T*>(tensors[blockIdx.y].weight);
		const int elements = tensors[blockIdx.y].elements;

		const T _scale = static_cast<T>(scale);

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] += _scale * weight[i];
	}

	struct nan_inf_tensors
	{
			const void *data;
			int32_t elements;
			uint32_t flag;
	};
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

		assert(cuda::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

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

		assert(cuda::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

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
		dim3 gridDim = cuda::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

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

	void cuda_radam_optimize(mlContext_t context, float scale, const mlTensor_t gradient, mlTensor_t weights, mlTensor_t momentum,
			mlTensor_t variance, float learning_rate, float beta1, float beta2, int step)
	{
		assert(step > 0);
		const int elements = volume(gradient);
		dim3 blockDim(256);

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 gridDim_x4 = cuda::gridSize<1024>(elements / 4, blockDim.x);
		dim3 gridDim_x1 = cuda::gridSize<1024>(elements, blockDim.x);
		switch (gradient.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 4 == 0)
					kernel_learn_radam<4> <<<gridDim_x4, blockDim, 0, stream>>>(scale, data<half>(gradient), data<float>(weights),
							data<float>(momentum), data<float>(variance), elements, learning_rate, beta1, beta2, step);
				else
					kernel_learn_radam<1> <<<gridDim_x1, blockDim, 0, stream>>>(scale, data<half>(gradient), data<float>(weights),
							data<float>(momentum), data<float>(variance), elements, learning_rate, beta1, beta2, step);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_learn_radam<4> <<<gridDim_x4, blockDim, 0, stream>>>(scale, data<float>(gradient), data<float>(weights),
							data<float>(momentum), data<float>(variance), elements, learning_rate, beta1, beta2, step);
				else
					kernel_learn_radam<1> <<<gridDim_x1, blockDim, 0, stream>>>(scale, data<float>(gradient), data<float>(weights),
							data<float>(momentum), data<float>(variance), elements, learning_rate, beta1, beta2, step);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	int cuda_is_nan_or_inf(mlContext_t context, const mlTensor_t tensor)
	{
		const int elements = volume(tensor);
		if (elements == 0)
			return 0;

		assert(ml::cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(uint32_t));

		uint32_t *workspace = cuda::Context::getWorkspace<uint32_t>(context);

		dim3 blockDim(256);
		dim3 gridDim((elements + 255) / 256);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (tensor.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 4 == 0)
					kernel_is_nan_or_inf_step_1<half, 4> <<<gridDim, blockDim, 0, stream>>>(workspace, data<half>(tensor), elements);
				else
					kernel_is_nan_or_inf_step_1<half, 1> <<<gridDim, blockDim, 0, stream>>>(workspace, data<half>(tensor), elements);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_is_nan_or_inf_step_1<float, 4> <<<gridDim, blockDim, 0, stream>>>(workspace, data<float>(tensor), elements);
				else
					kernel_is_nan_or_inf_step_1<float, 1> <<<gridDim, blockDim, 0, stream>>>(workspace, data<float>(tensor), elements);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

		kernel_is_nan_or_inf_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		uint32_t result = 0;
		cudaMemcpyAsync(&result, workspace, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result;
	}
	void cuda_l2_regularization(mlContext_t context, mlTensor_t gradient, const mlTensor_t param, float coefficient, float offset)
	{
		const int elements = volume(param);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, blockDim.x);

		switch (param.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 4 == 0)
					kernel_regularizer_l2<half, 4> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(data<half>(gradient),
							data<half>(param), coefficient, offset, elements);
				else
					kernel_regularizer_l2<half, 1> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(data<half>(gradient),
							data<half>(param), coefficient, offset, elements);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_regularizer_l2<float, 4> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(data<float>(gradient),
							data<float>(param), coefficient, offset, elements);
				else
					kernel_regularizer_l2<float, 1> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(data<float>(gradient),
							data<float>(param), coefficient, offset, elements);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_fused_radam_optimize(mlContext_t context, float scale, const mlTensor_t *gradient, mlTensor_t *weights, mlTensor_t *momentum,
			mlTensor_t *variance, float learning_rate, float beta1, float beta2, int step, int num_tensors)
	{
		assert(step > 0);
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = cuda::Context::getStream(context);

		radam_tensors *cpu_workspace = reinterpret_cast<radam_tensors*>(cuda::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			cpu_workspace[i].gradient = gradient[i].data;
			cpu_workspace[i].weight = data<float>(weights[i]);
			cpu_workspace[i].momentum = data<float>(momentum[i]);
			cpu_workspace[i].variance = data<float>(variance[i]);
			cpu_workspace[i].elements = volume(gradient[i]);
		}

		radam_tensors *device_workspace = cuda::Context::getWorkspace<radam_tensors>(context);
		cudaError_t status = cudaMemcpyAsync(device_workspace, cpu_workspace, sizeof(radam_tensors) * num_tensors, cudaMemcpyHostToDevice, stream);
		assert(status == cudaSuccess);

		dim3 blockDim(256);
		dim3 gridDim(32, num_tensors);
		switch (gradient[0].dtype)
		{
			case DTYPE_FLOAT16:
				kernel_fused_learn_radam<half> <<<gridDim, blockDim, 0, stream>>>(scale, device_workspace, learning_rate, beta1, beta2, step);
				break;
			case DTYPE_FLOAT32:
				kernel_fused_learn_radam<float> <<<gridDim, blockDim, 0, stream>>>(scale, device_workspace, learning_rate, beta1, beta2, step);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_fused_is_nan_or_inf(mlContext_t context, const mlTensor_t *tensor, int *result, int num_tensors)
	{
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = cuda::Context::getStream(context);

		nan_inf_tensors *cpu_workspace = reinterpret_cast<nan_inf_tensors*>(cuda::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			cpu_workspace[i].data = tensor[i].data;
			cpu_workspace[i].elements = volume(tensor[i]);
			cpu_workspace[i].flag = 0u;
		}

		nan_inf_tensors *device_workspace = cuda::Context::getWorkspace<nan_inf_tensors>(context);
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
	void cuda_fused_l2_regularization(mlContext_t context, mlTensor_t *gradient, const mlTensor_t *param, float scale, int num_tensors)
	{
		if (num_tensors <= 0)
			return;
		cudaStream_t stream = cuda::Context::getStream(context);

		l2_tensors *cpu_workspace = reinterpret_cast<l2_tensors*>(cuda::Context::getCpuWorkspace(context));
		for (int i = 0; i < num_tensors; i++)
		{
			cpu_workspace[i].gradient = gradient[i].data;
			cpu_workspace[i].weight = param[i].data;
			cpu_workspace[i].elements = volume(gradient[i]);
		}

		l2_tensors *device_workspace = cuda::Context::getWorkspace<l2_tensors>(context);
		cudaError_t status = cudaMemcpyAsync(device_workspace, cpu_workspace, sizeof(l2_tensors) * num_tensors, cudaMemcpyHostToDevice, stream);
		assert(status == cudaSuccess);

		dim3 blockDim(256);
		dim3 gridDim(32, num_tensors);
		switch (gradient[0].dtype)
		{
			case DTYPE_FLOAT16:
				kernel_fused_regularizer_l2<half> <<<gridDim, blockDim, 0, stream>>>(device_workspace, scale);
				break;
			case DTYPE_FLOAT32:
				kernel_fused_regularizer_l2<float> <<<gridDim, blockDim, 0, stream>>>(device_workspace, scale);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

