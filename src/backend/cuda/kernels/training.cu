/*
 * training.cu
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vectors/vectors.cuh"
#include "../vec/vec1f.cuh"
#include "../vec/vec4f.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
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
	using namespace vectors2;

//	__device__ float round_small_to_zero(float x)
//	{
//		return (fabsf(x) < 1.0e-6f) ? 0.0f : x;
//	}
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
		return (fabsf(x.x0) < 1.0e-6f) ? 0.0f : x.x0;
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

	__global__ void kernel_loss_gradient(float *gradient, const float *output, const float *target, const float *mask, int elements,
			float inv_batch_size)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			const float m = (mask == nullptr) ? 1.0f : mask[i];
			gradient[i] = m * inv_batch_size * (output[i] - target[i]);
		}
	}
	__global__ void kernel_value_head_loss_gradient(float *gradient, const float *output, const float *target, int first_dim, float inv_batch_size)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < first_dim; i += gridDim.x * blockDim.x)
		{
			const float mean = output[i * 2 + 0];
			const float variance = 1.0f; //output[i * 2 + 1];
			const float Q = target[i];

			gradient[i * 2 + 0] = inv_batch_size * 2.0f * (mean - Q) / variance;
			gradient[i * 2 + 1] = 0.0f; //inv_batch_size * (variance - square(mean - Q)) / square(variance);
		}
	}
	__global__ void kernel_CE_loss_step_1(float *workspace, const float *output, const float *target, const float *mask, int elements)
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
			const float m = (mask == nullptr) ? 1.0f : mask[i];
			acc += m * max(0.0f, cross_entropy(output[i], target[i]) - cross_entropy(target[i], target[i]));
		}
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = sum;
	}
	__global__ void kernel_MSE_loss_step_1(float *workspace, const float *output, const float *target, const float *mask, int elements)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			const float m = (mask == nullptr) ? 1.0f : mask[i];
			acc += m * square(output[i] - target[i]);
		}
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = sum;
	}
	__global__ void kernel_value_head_loss_step_1(float *workspace, const float *output, const float *target, int first_dim)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < first_dim; i += gridDim.x * blockDim.x)
		{
			const float mean = output[i * 2 + 0];
			const float variance = 1.0f; //output[i * 2 + 1];
			acc += std::log(variance) + square(mean - target[i]) / variance;
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

	template<int N>
	__global__ void kernel_learn_radam(float *weight, const float *gradient, float *momentum, float *variance, int elements, float learning_rate,
			float beta1, float beta2, int step, float weight_decay)
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
			vec<float, N> g(gradient + i);
			vec<float, N> m(momentum + i);
			vec<float, N> v(variance + i);

			m = beta1 * m + (1.0f - beta1) * g;
			v = beta2 * v + (1.0f - beta2) * square(g);

			vec<float, N> correction = 1.0f;
			if (p > 4.0f)
				correction = sqrt((1.0f - pow_beta2) / (v + 1.0e-8f)) * r;

			w = round_small_to_zero(w - learning_rate * (correction * m / (1.0f - pow_beta1) + weight_decay * w));

			m.store(momentum + i);
			v.store(variance + i);
			w.store(weight + i);
		}
	}

	template<int N>
	__global__ void kernel_regularizer_l2(float *gradient, const float *param, float scale, float offset, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = N * tid; i < elements; i += N * stride)
		{
			vec<float, N> g(gradient + i);
			vec<float, N> w(param + i);
			g += scale * (w - offset);
			g.store(gradient + i);
		}
	}
}

namespace ml
{
	float cuda_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target, const float *mask)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int length = volume(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_MSE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target),
				getPointer<float>(mask), length);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result / get_first_dim(shape);
	}
	void cuda_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, const float *mask,
			float weight)
	{
		cuda_cross_entropy_gradient(context, shape, gradient, output, target, mask, weight); // in this case both gradients are the same
	}
	float cuda_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target, const void *mask)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int length = volume(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_CE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target),
				getPointer<float>(mask), length);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result / get_first_dim(shape);
	}
	void cuda_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, const void *mask,
			float weight)
	{
		assert(output != nullptr);
		assert(target != nullptr);
		assert(gradient != nullptr);

		const int length = volume(shape);
		const float inv_batch_size = weight / get_first_dim(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_loss_gradient<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient), getPointer<float>(output), getPointer<float>(target),
				getPointer<float>(mask), length, inv_batch_size);
		assert(cudaGetLastError() == cudaSuccess);
	}
	float cuda_value_head_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int first_dim = get_first_dim(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(first_dim, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_value_head_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target), first_dim);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result / get_first_dim(shape);
	}
	void cuda_value_head_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		assert(output != nullptr);
		assert(target != nullptr);
		assert(gradient != nullptr);

		const int first_dim = get_first_dim(shape);
		const float inv_batch_size = weight / get_first_dim(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(first_dim, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_value_head_loss_gradient<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient), getPointer<float>(output),
				getPointer<float>(target), first_dim, inv_batch_size);
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2, int step, float weight_decay)
	{
		assert(weight != nullptr);
		assert(update != nullptr);
		assert(momentum != nullptr);
		assert(variance != nullptr);
		assert(step > 0);
		const int length = volume(shape);
		dim3 blockDim(256);

		cudaStream_t stream = cuda::Context::getStream(context);

		if (length % 4 == 0)
		{
			dim3 gridDim = cuda::gridSize<1024>(length / 4, blockDim.x);
			kernel_learn_radam<4> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(weight), getPointer<float>(update),
					getPointer<float>(momentum), getPointer<float>(variance), length, learning_rate, beta1, beta2, step, weight_decay);
		}
		else
		{
			dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
			kernel_learn_radam<1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(weight), getPointer<float>(update),
					getPointer<float>(momentum), getPointer<float>(variance), length, learning_rate, beta1, beta2, step, weight_decay);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset)
	{
		assert(gradient != nullptr);
		assert(param != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);

		if (length % 4 == 0)
		{
			dim3 gridDim = cuda::gridSize<1024>(length / 4, blockDim.x);
			kernel_regularizer_l2<4> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<float>(gradient),
					getPointer<float>(param), coefficient, offset, length);
		}
		else
		{
			dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
			kernel_regularizer_l2<1> <<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<float>(gradient),
					getPointer<float>(param), coefficient, offset, length);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

