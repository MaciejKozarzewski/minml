/*
 * softmax.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"

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
	__device__ vec<T, N> activation_forward(const vec<T, N> &x)
	{
		switch (ACT)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return x;
			case ml::ACTIVATION_SIGMOID:
				return vectors::sigmoid(x);
			case ml::ACTIVATION_TANH:
				return vectors::tanh(x);
			case ml::ACTIVATION_RELU:
				return vectors::relu(x);
			case ml::ACTIVATION_EXP:
				return vectors::exp(x);
		}
	}

	template<int ACT, typename T, int N>
	__device__ vec<T, N> activation_backward(const vec<T, N> &gradient, const vec<T, N> &input, const vec<T, N> &output)
	{
		const vec<T, N> one(1.0f);
		const vec<T, N> zero(0.0f);
		switch (ACT)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return gradient;
			case ml::ACTIVATION_SIGMOID:
				return gradient * output * (one - output);
			case ml::ACTIVATION_TANH:
				return gradient * (one - square(output));
			case ml::ACTIVATION_RELU:
				return select(output == zero, zero, gradient);
			case ml::ACTIVATION_EXP:
				return gradient * output;
		}
	}

	template<int ACT, typename T, int N>
	__global__ void kernel_activation_forward(float beta, T *output, float alpha, const T *input, int elements)
	{
		assert(elements % N == 0);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> x(input + i);
			vec<T, N> y = activation_forward<ACT>(x) * vec<T, N>(alpha);
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
			vec<T, N> dx = activation_backward<ACT>(dy, vec<T, N>(), y) * vec<T, N>(alpha);
			if (beta != 0.0f)
				dx += vec<T, N>(beta) * vec<T, N>(gradient_prev + i);
			dx.store(gradient_prev + i);
		}
	}

	template<int ACT, typename T, int N>
	__global__ void kernel_add_to_last_dim(float beta, T *output, float alpha, const T *input, const T *bias, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * N; j < last_dim; j += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> _bias(bias + j);
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int offset = i * last_dim + j;
				vec<T, N> tmp(input + offset);
				tmp = activation_forward<ACT>(tmp + _bias) * vec<T, N>(alpha);
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
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda::gridSize<1024>(elements, 256);

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
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda::gridSize<1024>(elements, 256);
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
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
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
			case ACTIVATION_EXP:
				dispatch_activation_backward<ACTIVATION_EXP>(context, alpha, dy, y, beta, dx);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
//	void cuda_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
//	{
//		assert(input != nullptr);
//		assert(output != nullptr);
//
//		switch (act)
//		{
//			case ACTIVATION_LINEAR:
//				if (output != input)
//					ml::cuda_memcpy_within_device(context, output, 0, input, 0, size_of(dtype) * volume(shape));
//				break;
//			case ACTIVATION_SIGMOID:
//				dispatch_activation_forward<ACTIVATION_SIGMOID>(context, dtype, shape, output, input);
//				break;
//			case ACTIVATION_TANH:
//				dispatch_activation_forward<ACTIVATION_TANH>(context, dtype, shape, output, input);
//				break;
//			case ACTIVATION_RELU:
//				dispatch_activation_forward<ACTIVATION_RELU>(context, dtype, shape, output, input);
//				break;
//			case ACTIVATION_GELU:
//				dispatch_activation_forward<ACTIVATION_GELU>(context, dtype, shape, output, input);
//				break;
//			case ACTIVATION_EXP:
//				dispatch_activation_forward<ACTIVATION_EXP>(context, dtype, shape, output, input);
//				break;
//		}
//		assert(cudaGetLastError() == cudaSuccess);
//	}
//	void cuda_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
//			mlActivationType_t act)
//	{
//		assert(gradient_prev != nullptr);
//		assert(gradient_next != nullptr);
//		assert(output != nullptr);
//
//		switch (act)
//		{
//			case ACTIVATION_LINEAR:
//				if (gradient_prev != gradient_next)
//					ml::cuda_memcpy_within_device(context, gradient_prev, 0, gradient_next, 0, sizeof(float) * volume(shape));
//				break;
//			case ACTIVATION_SIGMOID:
//				dispatch_activation_backward<ACTIVATION_SIGMOID>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
//				break;
//			case ACTIVATION_TANH:
//				dispatch_activation_backward<ACTIVATION_TANH>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
//				break;
//			case ACTIVATION_RELU:
//				dispatch_activation_backward<ACTIVATION_RELU>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
//				break;
//			case ACTIVATION_EXP:
//				dispatch_activation_backward<ACTIVATION_EXP>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
//				break;
//		}
//		assert(cudaGetLastError() == cudaSuccess);
//	}

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
			case ACTIVATION_EXP:
				dispatch_add_to_last_dim<ACTIVATION_EXP>(context, alpha, x, b, beta, y, act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
