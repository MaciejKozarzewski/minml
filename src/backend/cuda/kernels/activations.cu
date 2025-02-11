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

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors2;

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
				return vectors2::sigmoid(x);
			case ml::ACTIVATION_TANH:
				return vectors2::tanh(x);
			case ml::ACTIVATION_RELU:
				return vectors2::relu(x);
			case ml::ACTIVATION_GELU:
				return vectors2::approx_gelu(x);
			case ml::ACTIVATION_EXP:
				return vectors2::exp(x);
		}
	}

	template<int ACT, typename T>
	__device__ T activation_backward(T gradient, T input, T output)
	{
		switch (ACT)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return gradient;
			case ml::ACTIVATION_SIGMOID:
				return gradient * output * (1.0f - output);
			case ml::ACTIVATION_TANH:
				return gradient * (1.0f - output * output);
			case ml::ACTIVATION_RELU:
				return (output == 0.0f) ? 0.0f : gradient;
			case ml::ACTIVATION_GELU:
			{
				const T tmp = 1.0f / (1.0f + std::exp(-1.6849f * input));
				return gradient * (tmp + 1.6849f * input * tmp * (1.0f - tmp));
			}
			case ml::ACTIVATION_EXP:
				return gradient * output;
		}
	}

	template<int ACT, typename T, int N, typename U>
	__global__ void kernel_activation_forward(U *output, const U *input, int elements)
	{
		assert(elements % N == 0);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> x = load_vec<T, N>(input + i);
			const vec<T, N> y = activation_forward<ACT>(x);
			store_vec(output + i, y);
		}
	}
	template<int ACT, typename T>
	__global__ void kernel_activation_backward(T *gradient_prev, const T *gradient_next, const T *output, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient_prev[i] = activation_backward<ACT>(gradient_next[i], T { }, output[i]);
	}
	template<typename T>
	__global__ void kernel_gelu_backward(T *gradient_prev, const T *gradient_next, const T *input, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient_prev[i] = activation_backward<ml::ACTIVATION_GELU>(gradient_next[i], input[i], T { });
	}

	template<int ACT, typename T, int N, typename U>
	__global__ void kernel_add_to_last_dim(U *output, const U *input, const U *bias, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);

		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * N; j < last_dim; j += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> _bias = load_vec<T, N>(bias + j);
			for (int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				const int offset = i * last_dim + j;
				vec<T, N> tmp = (input == output) ? load_vec<T, N>(output + offset) : load_vec<T, N>(input + offset);
				tmp = activation_forward<ACT>(tmp + _bias);
				store_vec(output + offset, tmp);
			}
		}
	}

	template<int ACT>
	void dispatch_activation_forward(ml::mlContext_t context, ml::mlDataType_t dtype, ml::mlShape_t shape, void *output, const void *input)
	{
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda::gridSize<1024>(volume(shape), 256);
		const int elements = volume(shape);
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
				if (ml::cuda::has_fp16_math(context))
				{
					if (elements % 8 == 0)
						kernel_activation_forward<ACT, half, 8> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), elements);
					else
						kernel_activation_forward<ACT, half, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), elements);
				}
				else
				{
					if (elements % 4 == 0)
						kernel_activation_forward<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), elements);
					else
						kernel_activation_forward<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), elements);
				}
				break;
			case ml::DTYPE_FLOAT32:
				if (elements % 4 == 0)
					kernel_activation_forward<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), elements);
				else
					kernel_activation_forward<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), elements);
				break;
			case ml::DTYPE_FLOAT64:
				kernel_activation_forward<ACT, double, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output),
						ml::getPointer<double>(input), elements);
				break;
		}
	}
	template<int ACT>
	void dispatch_activation_backward(ml::mlContext_t context, ml::mlDataType_t dtype, ml::mlShape_t shape, void *gradient_prev,
			const void *gradient_next, const void *input, const void *output)
	{
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda::gridSize<1024>(volume(shape), 256);
		const int elements = volume(shape);
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
//				if (ml::cuda::has_fp16_math(context))
//					kernel_activation_backward<ACT, half, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(gradient_prev),
//							ml::getPointer<half>(gradient_next), ml::getPointer<half>(input), ml::getPointer<half>(output), elements);
//				else
//					kernel_activation_backward<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(gradient_prev),
//							ml::getPointer<half>(gradient_next), ml::getPointer<half>(input), ml::getPointer<half>(output), elements);
				break;
			case ml::DTYPE_FLOAT32:
				if (ACT == ml::ACTIVATION_GELU)
					kernel_gelu_backward <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(gradient_prev),
							ml::getPointer<float>(gradient_next), ml::getPointer<float>(input), elements);
				else
					kernel_activation_backward<ACT> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(gradient_prev),
							ml::getPointer<float>(gradient_next), ml::getPointer<float>(output), elements);
				break;
			case ml::DTYPE_FLOAT64:
				if (ACT == ml::ACTIVATION_GELU)
					kernel_gelu_backward <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(gradient_prev),
							ml::getPointer<double>(gradient_next), ml::getPointer<double>(input), elements);
				else
					kernel_activation_backward<ACT> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(gradient_prev),
							ml::getPointer<double>(gradient_next), ml::getPointer<double>(output), elements);
				break;
		}
	}

	template<int ACT>
	void dispatch_add_to_last_dim(ml::mlContext_t context, ml::mlDataType_t dtype, ml::mlShape_t shape, void *output, const void *input,
			const void *bias)
	{
		cudaStream_t stream = ml::cuda::Context::getStream(context);
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		dim3 blockDim(128);
		dim3 gridDim(ml::cuda::gridSize<128>(last_dim, blockDim.x), std::min(1024, first_dim));
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
				if (ml::cuda::has_fp16_math(context))
				{
					if (last_dim % 8 == 0)
						kernel_add_to_last_dim<ACT, half, 8> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), ml::getPointer<half>(bias), first_dim, last_dim);
					else
						kernel_add_to_last_dim<ACT, half, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), ml::getPointer<half>(bias), first_dim, last_dim);
				}
				else
				{
					if (last_dim % 4 == 0)
						kernel_add_to_last_dim<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), ml::getPointer<half>(bias), first_dim, last_dim);
					else
						kernel_add_to_last_dim<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(output),
								ml::getPointer<half>(input), ml::getPointer<half>(bias), first_dim, last_dim);
				}
				break;
			case ml::DTYPE_FLOAT32:
				if (last_dim % 4 == 0)
					kernel_add_to_last_dim<ACT, float, 4> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), ml::getPointer<float>(bias), first_dim, last_dim);
				else
					kernel_add_to_last_dim<ACT, float, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(output),
							ml::getPointer<float>(input), ml::getPointer<float>(bias), first_dim, last_dim);
				break;
			case ml::DTYPE_FLOAT64:
				kernel_add_to_last_dim<ACT, double, 1> <<<gridDim, blockDim, 0, stream>>>(ml::getPointer<double>(output),
						ml::getPointer<double>(input), ml::getPointer<double>(bias), first_dim, last_dim);
				break;
		}
	}
}

namespace ml
{
	void cuda_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(output != nullptr);

		switch (act)
		{
			case ACTIVATION_LINEAR:
				if (output != input)
					ml::cuda_memcpy_within_device(context, output, 0, input, 0, size_of(dtype) * volume(shape));
				break;
			case ACTIVATION_SIGMOID:
				dispatch_activation_forward<ACTIVATION_SIGMOID>(context, dtype, shape, output, input);
				break;
			case ACTIVATION_TANH:
				dispatch_activation_forward<ACTIVATION_TANH>(context, dtype, shape, output, input);
				break;
			case ACTIVATION_RELU:
				dispatch_activation_forward<ACTIVATION_RELU>(context, dtype, shape, output, input);
				break;
			case ACTIVATION_GELU:
				dispatch_activation_forward<ACTIVATION_GELU>(context, dtype, shape, output, input);
				break;
			case ACTIVATION_EXP:
				dispatch_activation_forward<ACTIVATION_EXP>(context, dtype, shape, output, input);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		assert(gradient_prev != nullptr);
		assert(gradient_next != nullptr);
		assert(output != nullptr);

		switch (act)
		{
			case ACTIVATION_LINEAR:
				if (gradient_prev != gradient_next)
					ml::cuda_memcpy_within_device(context, gradient_prev, 0, gradient_next, 0, sizeof(float) * volume(shape));
				break;
			case ACTIVATION_SIGMOID:
				dispatch_activation_backward<ACTIVATION_SIGMOID>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
				break;
			case ACTIVATION_TANH:
				dispatch_activation_backward<ACTIVATION_TANH>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
				break;
			case ACTIVATION_RELU:
				dispatch_activation_backward<ACTIVATION_RELU>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
				break;
			case ACTIVATION_EXP:
				dispatch_activation_backward<ACTIVATION_EXP>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, nullptr, output);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_gelu_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *input)
	{
		assert(gradient_prev != nullptr);
		assert(gradient_next != nullptr);
		assert(input != nullptr);

		dispatch_activation_backward<ACTIVATION_GELU>(context, DTYPE_FLOAT32, shape, gradient_prev, gradient_next, input, nullptr);
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
			mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(bias != nullptr);

		switch (act)
		{
			case ACTIVATION_LINEAR:
				dispatch_add_to_last_dim<ACTIVATION_LINEAR>(context, dtype, shape, output, input, bias);
				break;
			case ACTIVATION_SIGMOID:
				dispatch_add_to_last_dim<ACTIVATION_SIGMOID>(context, dtype, shape, output, input, bias);
				break;
			case ACTIVATION_TANH:
				dispatch_add_to_last_dim<ACTIVATION_TANH>(context, dtype, shape, output, input, bias);
				break;
			case ACTIVATION_RELU:
				dispatch_add_to_last_dim<ACTIVATION_RELU>(context, dtype, shape, output, input, bias);
				break;
			case ACTIVATION_GELU:
				dispatch_add_to_last_dim<ACTIVATION_GELU>(context, dtype, shape, output, input, bias);
				break;
			case ACTIVATION_EXP:
				dispatch_add_to_last_dim<ACTIVATION_EXP>(context, dtype, shape, output, input, bias);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
