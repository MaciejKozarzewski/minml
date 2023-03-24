/*
 * training.cu
 *
 *  Created on: Jan 4, 2023
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
#include <iostream>
#include <cassert>
#include <vector>

namespace
{
	template<typename T>
	__device__ bool is_power_of_2(T x)
	{
		return x > 0 && !(x & (x - 1));
	}

	__device__ float round_small_to_zero(float x)
	{
		return (fabsf(x) < 1.0e-6f) ? 0.0f : x;
	}
	__device__ float safe_log(float x)
	{
		return logf(1.0e-6f + x);
	}
	__device__ float cross_entropy(float output, float target)
	{
		return -target * safe_log(output) - (1.0f - target) * safe_log(1.0f - output);
	}
	__device__ float square(float x)
	{
		return x * x;
	}

	__global__ void kernel_loss_gradient(float *gradient, const float *output, const float *target, int elements, float inv_batch_size)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] = inv_batch_size * (output[i] - target[i]);
	}
	__global__ void kernel_loss_step_1(float *workspace, const float *output, const float *target, int elements, float inv_batch_size)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			acc += max(0.0f, cross_entropy(output[i], target[i]) - cross_entropy(target[i], target[i]));
		const float sum = cg::reduce(tile, acc, cg::plus<float>());
		if (threadIdx.x == 0)
			workspace[blockIdx.x] = sum * inv_batch_size;
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

	__global__ void kernel_learn_adam(float *weight, float *update, float *momentum, float *variance, int elements, float learning_rate, float beta1,
			float beta2)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			momentum[i] = momentum[i] * beta1 + update[i] * (1.0f - beta1);
			variance[i] = variance[i] * beta2 + square(update[i]) * (1.0f - beta2);
			const float tmp = -momentum[i] * learning_rate / sqrt(variance[i] + 1.0e-8f);
			weight[i] = round_small_to_zero(weight[i] + tmp);
			update[i] = 0.0f;
		}
	}

	__global__ void kernel_regularizer_l2(float *gradient, const float *param, float scale, float offset, int elements)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] += scale * (param[i] - offset);
	}

	template<int step>
	__global__ void kernel_sum_over_first_dim(float *dst, const float *src, int first_dim, int last_dim, float beta)
	{
		__shared__ float tmp[32][32];

		const int tid = blockIdx.x * 32 + threadIdx.x;
		if (tid < last_dim)
		{
			float result = 0.0f;
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				result += src[i * last_dim + tid];
			tmp[threadIdx.y][threadIdx.x] = result;
		}
		__syncthreads();

		for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
				tmp[threadIdx.y][threadIdx.x] += tmp[i + threadIdx.y][threadIdx.x];
			__syncthreads();
		}

		__syncthreads();
		if (threadIdx.y == 0 && tid < last_dim)
		{
			if (step == 1) // write to temporary storage array
				dst[blockIdx.y * last_dim + tid] = tmp[0][threadIdx.x];
			if (step == 2) // write to final destination
				dst[tid] = beta * dst[tid] + tmp[0][threadIdx.x];
		}
	}

	__global__ void kernel_add_tensors(float *dst, const float *src0, const float *src1, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] = src0[i] + src1[i];
	}
	__global__ void kernel_add_tensors(float *dst, const float *src, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] += src[i];
	}

	__global__ void kernel_emulate_low_precision(uint32_t *dst, const uint32_t *src, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] = src[i] & 0xFFFFF000u;
	}
}

namespace ml
{
	void cuda_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src)
	{
		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);

		kernel_emulate_low_precision<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<uint32_t>(dst),
				getPointer<uint32_t>(src), length);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_add_tensors(mlContext_t context, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (dst == src1)
		{ // in place addition
			kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src2), length);
		}
		else
		{
			kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src1), getPointer<float>(src2), length);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta)
	{
		assert(dst != nullptr);
		assert(src != nullptr);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= last_dim * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) 256, cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 blockDim(32, 32);
		dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);
		dim3 gridDim2((last_dim + 31) / 32);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_sum_over_first_dim<1> <<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<float>(src), first_dim, last_dim, beta);
		assert(cudaGetLastError() == cudaSuccess);
		kernel_sum_over_first_dim<2> <<<gridDim2, blockDim, 0, stream>>>(getPointer<float>(dst), workspace, workspace_first_dim, last_dim, beta);
		assert(cudaGetLastError() == cudaSuccess);
	}
	float cuda_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int length = volume(shape);
		const float inv_batch_size = 1.0f / get_first_dim(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target), length, inv_batch_size);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result;
	}
	void cuda_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
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
				length, inv_batch_size);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_adam_optimize(mlContext_t context, mlShape_t shape, void *weight, void *update, void *momentum, void *variance, float learning_rate,
			float beta1, float beta2)
	{
		assert(weight != nullptr);
		assert(update != nullptr);
		assert(momentum != nullptr);
		assert(variance != nullptr);
		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_learn_adam<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(weight), getPointer<float>(update), getPointer<float>(momentum),
				getPointer<float>(variance), length, learning_rate, beta1, beta2);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset)
	{
		assert(gradient != nullptr);
		assert(param != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);

		kernel_regularizer_l2<<<gridDim, blockDim, 0, cuda::Context::getStream(context)>>>(getPointer<float>(gradient), getPointer<float>(param),
				coefficient, offset, length);
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

