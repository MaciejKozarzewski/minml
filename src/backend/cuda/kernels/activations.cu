/*
 * softmax.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "activations.cuh"
#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"
#include "../helpers/indexers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;

	template<typename T>
	__device__ bool is_power_of_2(T x)
	{
		return x > 0 && !(x & (x - 1));
	}

	template<int ACT, typename T, int N>
	__global__ void kernel_activation_forward(float beta, T *output, float alpha, const T *input, int elements)
	{
		assert(elements % N == 0);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> x(input + i);
			vec<T, N> y = activation_forward(static_cast<ml::mlActivationType_t>(ACT), x) * vec<T, N>(alpha);
			if (beta != 0.0f)
				y += vec<T, N>(beta) * vec<T, N>(output + i);
			y.store(output + i);
		}
	}
	template<int ACT, typename T, int N>
	__global__ void kernel_activation_backward(float beta, T *gradient_prev, float alpha, const T *gradient_next, const T *output, int elements)
	{
		assert(elements % N == 0);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> dy(gradient_next + i);
			const vec<T, N> y(output + i);
			vec<T, N> dx = activation_backward(static_cast<ml::mlActivationType_t>(ACT), dy, vec<T, N>(), y) * vec<T, N>(alpha);
			if (beta != 0.0f)
				dx += vec<T, N>(beta) * vec<T, N>(gradient_prev + i);
			dx.store(gradient_prev + i);
		}
	}

	template<int ACT, typename T, int N>
	__global__ void kernel_add_to_last_dim(float beta, T *output, float alpha, const T *input, const T *bias, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		const Indexer<1> bias_indexer(last_dim);
		const Indexer<2> in_out_indexer(first_dim, last_dim);

		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * N; j < last_dim; j += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> _bias(bias + bias_indexer.at(j));
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int offset = in_out_indexer.at(i, j);
				vec<T, N> tmp(input + offset);
				tmp = activation_forward(static_cast<ml::mlActivationType_t>(ACT), tmp + _bias) * vec<T, N>(alpha);
				if (beta != 0.0f)
					tmp += vec<T, N>(beta) * vec<T, N>(output + offset);
				tmp.store(output + offset);
			}
		}
	}

	using namespace ml;
	template<int ACT>
	void dispatch_activation_forward(mlContext_t context, float alpha, const mlTensor_t &x, float beta, mlTensor_t &y)
	{
		const int elements = volume(x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, 256);

		switch (x.dtype)
		{
			case ml::DTYPE_FLOAT16:
				if (elements % 8 == 0)
					kernel_activation_forward<ACT, half, 8> <<<gridDim, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), elements);
				else
					kernel_activation_forward<ACT, half, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x), elements);
				break;
			case ml::DTYPE_FLOAT32:
				if (elements % 4 == 0)
					kernel_activation_forward<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							elements);
				else
					kernel_activation_forward<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							elements);
				break;
			case ml::DTYPE_FLOAT64:
				kernel_activation_forward<ACT, double, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<double>(y), alpha, data<double>(x), elements);
				break;
		}
	}
	template<int ACT>
	void dispatch_activation_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t y, float beta, mlTensor_t dx)
	{
		const int elements = volume(dx);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, 256);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
				if (elements % 8 == 0)
					kernel_activation_backward<ACT, half, 8> <<<gridDim, blockDim, 0, stream>>>(beta, data<half>(dx), alpha, data<half>(dy),
							data<half>(y), elements);
				else
					kernel_activation_backward<ACT, half, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<half>(dx), alpha, data<half>(dy),
							data<half>(y), elements);
				break;
			case DTYPE_FLOAT32:
				if (elements % 4 == 0)
					kernel_activation_backward<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(beta, data<float>(dx), alpha, data<float>(dy),
							data<float>(y), elements);
				else
					kernel_activation_backward<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<float>(dx), alpha, data<float>(dy),
							data<float>(y), elements);
				break;
			case DTYPE_FLOAT64:
				kernel_activation_backward<ACT, double, 1> <<<gridDim, blockDim, 0, stream>>>(beta, data<double>(dx), alpha, data<double>(dy),
						data<double>(y), elements);
				break;
		}
	}

	template<int ACT>
	void dispatch_add_to_last_dim(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t b, float beta, mlTensor_t y,
			mlActivationType_t act)
	{
		assert(same_shape(x, y));

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(get_last_dim(b) == last_dim);
		dim3 blockDim(32, 8);
		dim3 gridDim_x1((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8));
		dim3 gridDim_x4((last_dim + 127) / 128, std::min(1024, (first_dim + 7) / 8));

		switch (x.dtype)
		{
			case ml::DTYPE_FLOAT16:
				if (last_dim % 4 == 0)
					kernel_add_to_last_dim<ACT, half, 4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(b), first_dim, last_dim);
				else
					kernel_add_to_last_dim<ACT, half, 1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(b), first_dim, last_dim);
				break;
			case ml::DTYPE_FLOAT32:
				if (last_dim % 4 == 0)
					kernel_add_to_last_dim<ACT, float, 4> <<<gridDim_x4, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(b), first_dim, last_dim);
				else
					kernel_add_to_last_dim<ACT, float, 1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(b), first_dim, last_dim);
				break;
			case ml::DTYPE_FLOAT64:
				kernel_add_to_last_dim<ACT, double, 1> <<<gridDim_x1, blockDim, 0, stream>>>(beta, data<double>(y), alpha, data<double>(x),
						data<double>(b), first_dim, last_dim);
				break;
		}
	}
}

namespace ml
{
	void cuda_activation_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				if (x.data != y.data)
					ml::cuda_memcpy_within_device(context, data<uint8_t>(y), 0, data<uint8_t>(x), 0, size_of(x.dtype) * volume(x));
				break;
			case ACTIVATION_SIGMOID:
				dispatch_activation_forward<ACTIVATION_SIGMOID>(context, alpha, x, beta, y);
				break;
			case ACTIVATION_TANH:
				dispatch_activation_forward<ACTIVATION_TANH>(context, alpha, x, beta, y);
				break;
			case ACTIVATION_RELU:
				dispatch_activation_forward<ACTIVATION_RELU>(context, alpha, x, beta, y);
				break;
			case ACTIVATION_LEAKY_RELU:
				dispatch_activation_forward<ACTIVATION_LEAKY_RELU>(context, alpha, x, beta, y);
				break;
			case ACTIVATION_EXP:
				dispatch_activation_forward<ACTIVATION_EXP>(context, alpha, x, beta, y);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_activation_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t y, float beta, mlTensor_t dx,
			mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				if (dx.data != dy.data)
					ml::cuda_memcpy_within_device(context, data<uint8_t>(dx), 0, data<uint8_t>(dy), 0, size_of(dx.dtype) * volume(dx));
				break;
			case ACTIVATION_SIGMOID:
				dispatch_activation_backward<ACTIVATION_SIGMOID>(context, alpha, dy, y, beta, dx);
				break;
			case ACTIVATION_TANH:
				dispatch_activation_backward<ACTIVATION_TANH>(context, alpha, dy, y, beta, dx);
				break;
			case ACTIVATION_RELU:
				dispatch_activation_backward<ACTIVATION_RELU>(context, alpha, dy, y, beta, dx);
				break;
			case ACTIVATION_LEAKY_RELU:
				dispatch_activation_backward<ACTIVATION_LEAKY_RELU>(context, alpha, dy, y, beta, dx);
				break;
			case ACTIVATION_EXP:
				dispatch_activation_backward<ACTIVATION_EXP>(context, alpha, dy, y, beta, dx);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_add_bias_act(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t b, float beta, mlTensor_t y, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				dispatch_add_to_last_dim<ACTIVATION_LINEAR>(context, alpha, x, b, beta, y, act);
				break;
			case ACTIVATION_SIGMOID:
				dispatch_add_to_last_dim<ACTIVATION_SIGMOID>(context, alpha, x, b, beta, y, act);
				break;
			case ACTIVATION_TANH:
				dispatch_add_to_last_dim<ACTIVATION_TANH>(context, alpha, x, b, beta, y, act);
				break;
			case ACTIVATION_RELU:
				dispatch_add_to_last_dim<ACTIVATION_RELU>(context, alpha, x, b, beta, y, act);
				break;
			case ACTIVATION_LEAKY_RELU:
				dispatch_add_to_last_dim<ACTIVATION_LEAKY_RELU>(context, alpha, x, b, beta, y, act);
				break;
			case ACTIVATION_EXP:
				dispatch_add_to_last_dim<ACTIVATION_EXP>(context, alpha, x, b, beta, y, act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
