/*
 * learnable_scaling.cu
 *
 *  Created on: Feb 24, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "activations.cuh"
#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../helpers/misc.cuh"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;
	using namespace ml;

	template<typename T, int N>
	__global__ void kernel_learnable_scaling_forward(float beta, T *output, float alpha, const T *input, const T *weights, int first_dim,
			int last_dim, mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32 && blockDim.y == 8);
		const Indexer<1> scale_indexer(last_dim);
		const Indexer<2> tensor_indexer(first_dim, last_dim);
		const int dim0_index = blockIdx.y * blockDim.y + threadIdx.y;
		const int dim1_index = N * (blockIdx.x * blockDim.x + threadIdx.x);

		if (dim1_index < last_dim)
		{
			const vec<T, N> scale = activation_forward(act, vec<T, N>(weights + scale_indexer.at(dim1_index)));
			for (int i = dim0_index; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int idx = tensor_indexer.at(i, dim1_index);
				const vec<T, N> x(input + idx);
				vec<T, N> y = x * scale * vec<T, N>(alpha);
				if (beta != 0.0f)
					y += vec<T, N>(output + idx) * vec<T, N>(beta);
				y.store(output + idx);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_learnable_scaling_backward(float beta_prev, T *gradient_prev, T *gradient_weights, float alpha, const T *gradient_next,
			const T *input, const T *weights, int first_dim, int last_dim, mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32 && blockDim.y == 8);
		__shared__ vec<T, N> workspace[8][32];

		const int dim0_index = blockIdx.y * blockDim.y + threadIdx.y;
		const int dim1_index = N * (blockIdx.x * blockDim.x + threadIdx.x);

		const Indexer<1> scale_indexer(last_dim);
		if (dim1_index < last_dim)
		{
			const Indexer<2> tensor_indexer(first_dim, last_dim);

			vec<T, N> scale_gradient = vec<T, N>(0.0f);
			const vec<T, N> scale = activation_forward(act, vec<T, N>(weights + scale_indexer.at(dim1_index)));
			for (int i = dim0_index; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int tmp = tensor_indexer.at(i, dim1_index);
				const vec<T, N> gradient(gradient_next + tmp);
				const vec<T, N> inp(input + tmp);
				const vec<T, N> out = input * scale;
				scale_gradient += activation_backward(act, gradient, inp, out) * vec<T, N>(input + tmp);

				vec<T, N> dx = vec<T, N>(alpha) * scale * gradient;
				if (beta_prev != 0.0f)
					dx += vec<T, N>(beta_prev) * vec<T, N>(gradient_prev + tmp);
				dx.store(gradient_prev + tmp);
			}
			workspace[threadIdx.y][threadIdx.x] = scale_gradient;
		}

		__syncthreads();
		for (int i = blockDim.y / 2; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				workspace[threadIdx.y][threadIdx.x] += workspace[threadIdx.y + i][threadIdx.x];
			__syncthreads();
		}

		if (threadIdx.y == 0 && dim1_index < last_dim)
		{
			const vec<T, N> dscales = vec<T, N>(alpha) * workspace[0][threadIdx.x];
			atomic_add(gradient_weights + scale_indexer.at(dim0_index), dscales);
		}
	}

}

namespace ml
{
	void cuda_learnable_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, mlActivationType_t act, const mlTensor_t w, float beta,
			mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(same_shape(x, y));
		assert(w.rank == 1);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(last_dim == w.dim[0]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim_x1((last_dim + 31) / 32, std::min(1024, first_dim));
		dim3 gridDim_x4((last_dim + 127) / 128, std::min(1024, first_dim));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_learnable_scaling_forward<half, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(w), first_dim, last_dim, act);
				else
					kernel_learnable_scaling_forward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(w), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_learnable_scaling_forward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(w), first_dim, last_dim, act);
				else
					kernel_learnable_scaling_forward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(w), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT64:
				kernel_learnable_scaling_forward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						data<double>(w), first_dim, last_dim, act);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_learnable_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, mlActivationType_t act,
			const mlTensor_t w, float beta_dx, mlTensor_t dx, float beta_dw, mlTensor_t dw)
	{
		assert(x.rank == 4);
		assert(same_shape(dx, dy));
		assert(same_shape(dx, x));
		assert(w.rank == 1);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(last_dim == w.dim[0]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim_x1((last_dim + 31) / 32, std::min(1024, first_dim));
		dim3 gridDim_x4((last_dim + 127) / 128, std::min(1024, first_dim));

		cuda_scale_tensor(context, beta_dw, dw);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				if (last_dim % 4 == 0)
					kernel_learnable_scaling_backward<half, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta_dx, data<half>(dx), data<half>(dw), alpha,
							data<half>(dy), data<half>(x), data<half>(w), first_dim, last_dim, act);
				else
					kernel_learnable_scaling_backward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<half>(dx), data<half>(dw), alpha,
							data<half>(dy), data<half>(x), data<half>(w), first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT32:
				if (last_dim % 4 == 0)
					kernel_learnable_scaling_backward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta_dx, data<float>(dx), data<float>(dw),
							alpha, data<float>(dy), data<float>(x), data<float>(w), first_dim, last_dim, act);
				else
					kernel_learnable_scaling_backward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<float>(dx), data<float>(dw),
							alpha, data<float>(dy), data<float>(x), data<float>(w), first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT64:
				kernel_learnable_scaling_backward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<double>(dx), data<double>(dw),
						alpha, data<double>(dy), data<double>(x), data<double>(w), first_dim, last_dim, act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

