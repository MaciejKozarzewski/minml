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

	__device__ float round_small_to_zero(float x)
	{
		return (fabsf(x) < 1.0e-6f) ? 0.0f : x;
	}
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

	__global__ void kernel_loss_gradient(float *gradient, const float *output, const float *target, int elements, float inv_batch_size)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] = inv_batch_size * (output[i] - target[i]);
	}
	__global__ void kernel_CE_loss_step_1(float *workspace, const float *output, const float *target, int elements)
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
			workspace[blockIdx.x] = sum;
	}
	__global__ void kernel_MSE_loss_step_1(float *workspace, const float *output, const float *target, int elements)
	{
		assert(blockDim.x == 256);
		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		float acc = 0.0f;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			acc += square(output[i] - target[i]);
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
			float beta1, float beta2, int step)
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

			w = round_small_to_zero(w - learning_rate * correction * m / (1.0f - pow_beta1));

			m.store(momentum + i);
			v.store(variance + i);
			w.store(weight + i);

//			momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * gradient[i];
//			variance[i] = beta2 * variance[i] + (1.0f - beta2) * square(gradient[i]);
//
//			float correction = 1.0f;
//			if (p > 4.0f)
//			{
//				const float l = std::sqrt((1.0f - pow_beta2) / (variance[i] + 1.0e-8f));
//				const float r = std::sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));
//				correction = l * r;
//			}
//
//			const float m_dash = momentum[i] / (1.0f - pow_beta1);
//			const float tmp = -learning_rate * m_dash * correction;
//			weight[i] = round_small_to_zero(weight[i] + tmp);
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

	template<int step>
	__global__ void kernel_sum_over_first_dim_old(float *dst, const float *src, int first_dim, int last_dim, float beta)
	{
		__shared__ float workspace[32][33];

		const int tid = blockIdx.x * 32 + threadIdx.x;
		float local_sum = 0.0f;
		if (tid < last_dim)
		{
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				local_sum += src[i * last_dim + tid];
			workspace[threadIdx.y][threadIdx.x] = local_sum;
		}
		__syncthreads();
		local_sum = workspace[threadIdx.x][threadIdx.y];
		for (int k = 16; k >= 1; k /= 2)
			local_sum += __shfl_xor_sync(0xffffffff, local_sum, k);

		__syncthreads();
		workspace[0][threadIdx.y] = local_sum;

		__syncthreads();
		if (threadIdx.y == 0 && tid < last_dim)
		{
			if (step == 1) // write to temporary storage array
				dst[blockIdx.y * last_dim + tid] = workspace[0][threadIdx.x];
			if (step == 2) // write to final destination
			{
				if (beta == 0.0f)
					dst[tid] = workspace[0][threadIdx.x];
				else
					dst[tid] = beta * dst[tid] + workspace[0][threadIdx.x];
			}
		}
	}

	template<int Step>
	__global__ void kernel_sum_over_first_dim_vect(float *dst, const float *src, int first_dim, int last_dim, float beta)
	{
		assert(last_dim % 4 == 0);
		__shared__ float workspace[32][128 + 1];

		const int first_dim_idx = 32 * blockIdx.y + threadIdx.y;
		const int last_dim_idx = 4 * (32 * blockIdx.x + threadIdx.x);
		vec4f local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
			{
				const vec4f tmp(src + i * last_dim + last_dim_idx);
				local_sum += tmp;
			}
			workspace[threadIdx.y][4 * threadIdx.x + 0] = local_sum.x0;
			workspace[threadIdx.y][4 * threadIdx.x + 1] = local_sum.x1;
			workspace[threadIdx.y][4 * threadIdx.x + 2] = local_sum.x2;
			workspace[threadIdx.y][4 * threadIdx.x + 3] = local_sum.x3;
		}
		__syncthreads();
		local_sum.x0 = workspace[threadIdx.x][4 * threadIdx.y + 0];
		local_sum.x1 = workspace[threadIdx.x][4 * threadIdx.y + 1];
		local_sum.x2 = workspace[threadIdx.x][4 * threadIdx.y + 2];
		local_sum.x3 = workspace[threadIdx.x][4 * threadIdx.y + 3];

		for (int k = 16; k >= 1; k /= 2)
		{
			local_sum.x0 += __shfl_xor_sync(0xffffffff, local_sum.x0, k);
			local_sum.x1 += __shfl_xor_sync(0xffffffff, local_sum.x1, k);
			local_sum.x2 += __shfl_xor_sync(0xffffffff, local_sum.x2, k);
			local_sum.x3 += __shfl_xor_sync(0xffffffff, local_sum.x3, k);
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			workspace[0][4 * threadIdx.y + 0] = local_sum.x0;
			workspace[0][4 * threadIdx.y + 1] = local_sum.x1;
			workspace[0][4 * threadIdx.y + 2] = local_sum.x2;
			workspace[0][4 * threadIdx.y + 3] = local_sum.x3;
		}
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
		{
			vec4f tmp(workspace[0] + 4 * threadIdx.x);
			if (Step == 1) // write to temporary storage array
			{
				const int idx = blockIdx.y * last_dim + last_dim_idx;
				tmp.store(dst + idx);
			}
			if (Step == 2) // write to final destination
			{
				if (beta != 0.0f)
				{
					const vec4f y(dst + last_dim_idx);
					tmp += beta * y;
				}
				tmp.store(dst + last_dim_idx);
			}
		}
	}
	template<int Step>
	__global__ void kernel_sum_over_first_dim(float *dst, const float *src, int first_dim, int last_dim, float beta)
	{
		__shared__ float workspace[32][32 + 1];

		const int first_dim_idx = 32 * blockIdx.y + threadIdx.y;
		const int last_dim_idx = 32 * blockIdx.x + threadIdx.x;
		vec1f local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
			{
				const vec1f tmp(src + i * last_dim + last_dim_idx);
				local_sum += tmp;
			}
			workspace[threadIdx.y][threadIdx.x + 0] = local_sum.x0;
		}
		__syncthreads();
		local_sum.x0 = workspace[threadIdx.x][threadIdx.y + 0];

		for (int k = 16; k >= 1; k /= 2)
			local_sum.x0 += __shfl_xor_sync(0xffffffff, local_sum.x0, k);
		__syncthreads();
		if (threadIdx.x == 0)
			workspace[0][threadIdx.y + 0] = local_sum.x0;
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
		{
			vec1f tmp(workspace[0] + threadIdx.x);
			if (Step == 1) // write to temporary storage array
			{
				const int idx = blockIdx.y * last_dim + last_dim_idx;
				tmp.store(dst + idx);
			}
			if (Step == 2) // write to final destination
			{
				if (beta != 0.0f)
				{
					const vec1f y(dst + last_dim_idx);
					tmp += beta * y;
				}
				tmp.store(dst + last_dim_idx);
			}
		}
	}

	template<typename T>
	__global__ void kernel_multiply_tensors(T *dst, const T *src0, const T *src1, int elements)
	{
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); i < elements; i += gridDim.x * blockDim.x * vector_length<T>())
		{
			const int tmp = elements - i;
			const Vector<T> x0(src0 + i, tmp);
			const Vector<T> x1(src1 + i, tmp);
			const Vector<T> y = x0 * x1;
			y.store(dst + i, tmp);
		}
	}
	template<typename T>
	__global__ void kernel_multiply_tensors(T *dst, const T *src, int elements)
	{
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); i < elements; i += gridDim.x * blockDim.x * vector_length<T>())
		{
			const int tmp = elements - i;
			const Vector<T> x(src + i, tmp);
			const Vector<T> y = Vector<T>(dst + i, tmp) * x;
			y.store(dst + i, tmp);
		}
	}

	template<typename T>
	__global__ void kernel_add_tensors(T *dst, const T *src0, const T *src1, int elements)
	{
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); i < elements; i += gridDim.x * blockDim.x * vector_length<T>())
		{
			const int tmp = elements - i;
			const Vector<T> x0(src0 + i, tmp);
			const Vector<T> x1(src1 + i, tmp);
			const Vector<T> y = x0 + x1;
			y.store(dst + i, tmp);
		}
	}
	template<typename T>
	__global__ void kernel_add_tensors(T *dst, const T *src, int elements)
	{
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); i < elements; i += gridDim.x * blockDim.x * vector_length<T>())
		{
			const int tmp = elements - i;
			const Vector<T> x(src + i, tmp);
			const Vector<T> y = Vector<T>(dst + i, tmp) + x;
			y.store(dst + i, tmp);
		}
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
	void cuda_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
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
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src2), length);
					break;
				case DTYPE_FLOAT32:
					kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src2), length);
					break;
			}
		}
		else
		{
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src1), getPointer<half>(src2),
							length);
					break;
				case DTYPE_FLOAT32:
					kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src1),
							getPointer<float>(src2), length);
					break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
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
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src2), length);
					break;
				case DTYPE_FLOAT32:
					kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src2), length);
					break;
			}
		}
		else
		{
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src1), getPointer<half>(src2),
							length);
					break;
				case DTYPE_FLOAT32:
					kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src1), getPointer<float>(src2),
							length);
					break;
			}
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
		cudaStream_t stream = cuda::Context::getStream(context);

		if (last_dim % 4 == 0)
		{
			dim3 gridDim1((last_dim + 127) / 128, workspace_first_dim);
			dim3 gridDim2((last_dim + 127) / 128);
			kernel_sum_over_first_dim_vect<1> <<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<float>(src), first_dim, last_dim, beta);
			assert(cudaGetLastError() == cudaSuccess);
			kernel_sum_over_first_dim_vect<2> <<<gridDim2, blockDim, 0, stream>>>(getPointer<float>(dst), workspace, workspace_first_dim, last_dim,
					beta);
			assert(cudaGetLastError() == cudaSuccess);
		}
		else
		{
			dim3 gridDim1((last_dim + 31) / 32, workspace_first_dim);
			dim3 gridDim2((last_dim + 31) / 32);
			kernel_sum_over_first_dim<1> <<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<float>(src), first_dim, last_dim, beta);
			assert(cudaGetLastError() == cudaSuccess);
			kernel_sum_over_first_dim<2> <<<gridDim2, blockDim, 0, stream>>>(getPointer<float>(dst), workspace, workspace_first_dim, last_dim, beta);
			assert(cudaGetLastError() == cudaSuccess);
		}
	}
	float cuda_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int length = volume(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_MSE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target), length);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result / get_first_dim(shape);
	}
	void cuda_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		cuda_cross_entropy_gradient(context, shape, gradient, output, target, weight); // in this case both gradients are the same
	}
	float cuda_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int length = volume(shape);

		assert(cuda::Context::getWorkspaceSize(context) >= 4096 * sizeof(float));

		float *workspace = cuda::Context::getWorkspace<float>(context);

		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<4096>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_CE_loss_step_1<<<gridDim, blockDim, 0, stream>>>(workspace, getPointer<float>(output), getPointer<float>(target), length);
		assert(cudaGetLastError() == cudaSuccess);

		kernel_loss_step_2<<<1, blockDim, 0, stream>>>(workspace, gridDim.x);
		assert(cudaGetLastError() == cudaSuccess);

		float result = 0.0f;
		cudaMemcpyAsync(&result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaError_t status = cudaStreamSynchronize(stream);
		assert(status == cudaSuccess);
		return result / get_first_dim(shape);
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
	void cuda_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2, int step)
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
					getPointer<float>(momentum), getPointer<float>(variance), length, learning_rate, beta1, beta2, step);
		}
		else
		{
			dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
			kernel_learn_radam<1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(weight), getPointer<float>(update),
					getPointer<float>(momentum), getPointer<float>(variance), length, learning_rate, beta1, beta2, step);
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

