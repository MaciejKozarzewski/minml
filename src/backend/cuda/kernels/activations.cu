/*
 * softmax.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"

#include "../vectors/vectors.cuh"
#include "../helpers/tensor_wrappers.cuh"

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

	template<typename T>
	__global__ void kernel_softmax_3_channels(T *output, const T *input, int first_dim)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < first_dim)
		{
			float x0 = static_cast<float>(input[idx * 3 + 0]);
			float x1 = static_cast<float>(input[idx * 3 + 1]);
			float x2 = static_cast<float>(input[idx * 3 + 2]);

			const float max_value = max(x0, max(x1, x2));
			x0 = exp(x0 - max_value);
			x1 = exp(x1 - max_value);
			x2 = exp(x2 - max_value);

			const float inv_sum = 1.0f / (x0 + x1 + x2);

			output[idx * 3 + 0] = static_cast<T>(x0 * inv_sum);
			output[idx * 3 + 1] = static_cast<T>(x1 * inv_sum);
			output[idx * 3 + 2] = static_cast<T>(x2 * inv_sum);
		}
	}

	template<typename T>
	__global__ void kernel_softmax_generic(T *output, const T *input, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 128 > tile = cg::tiled_partition<128>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float max_value = -1e+32f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = static_cast<float>(input[i * last_dim + j]);
				max_value = max(max_value, workspace[j]);
			}
			const float shift = cg::reduce(tile, max_value, cg::greater<float>());

			float partial_sum = 0.0f;
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = exp(workspace[j] - shift);
				partial_sum += workspace[j];
			}
			const float inv_sum = 1.0f / cg::reduce(tile, partial_sum, cg::plus<float>());
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
				output[i * last_dim + j] = static_cast<T>(workspace[j] * inv_sum);
		}
	}

	template<typename T>
	__global__ void kernel_sigmoid_forward(T *output, const T *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
		{
			Vector<T> tmp(input + i, length - i);
			tmp = vector_one<T>() / (vector_one<T>() + vectors::exp(-tmp));
			tmp.store(output + i, length - i);
		}
	}
	__global__ void kernel_sigmoid_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			gradient_prev[i] = gradient_next[i] * output[i] * (1.0f - output[i]);
	}

	template<typename T>
	__global__ void kernel_tanh_forward(T *output, const T *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
		{
			Vector<T> tmp(input + i, length - i);
			tmp = vectors::tanh(tmp);
			tmp.store(output + i, length - i);
		}
	}
	__global__ void kernel_tanh_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			gradient_prev[i] = gradient_next[i] * (1.0f - output[i]) * (1.0f + output[i]);
	}

	template<typename T>
	__global__ void kernel_relu_forward(T *output, const T *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
		{
			Vector<T> tmp(input + i, length - i);
			tmp = max(vector_zero<T>(), tmp);
			tmp.store(output + i, length - i);
		}
	}
	__global__ void kernel_relu_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			gradient_prev[i] = (output[i] == 0.0f) ? 0.0f : gradient_next[i];
	}

	template<typename T>
	__global__ void kernel_add_to_last_dim(T *output, const T *input, const T *bias, int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);

		ConstTensorWrapper<1, T> bias_wrapper(bias, last_dim);

		ConstTensorWrapper<2, T> input_wrapper(input, first_dim, last_dim);
		TensorWrapper<2, T> output_wrapper(output, first_dim, last_dim);
		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); j < last_dim; j += gridDim.x * blockDim.x * vector_length<T>())
		{
			Vector<T> _bias = bias_wrapper.load(j);
			for (int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				Vector<T> tmp = input_wrapper.load(i, j);
				tmp += _bias;
				if (act == ml::ACTIVATION_RELU)
					tmp = vectors::max(vector_zero<T>(), tmp);
				if (act == ml::ACTIVATION_TANH)
					tmp = vectors::tanh(tmp);
				if (act == ml::ACTIVATION_SIGMOID)
					tmp = vector_one<T>() / (vector_one<T>() + vectors::exp(-tmp));
				output_wrapper.store(tmp, i, j);
			}
		}
	}
}

namespace ml
{
	void cuda_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(output != nullptr);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (output != input)
					ml::cuda_memcpy_within_device(context, output, 0, input, 0, size_of(dtype) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				switch (dtype)
				{
					case DTYPE_FLOAT16:
						kernel_sigmoid_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), volume(shape));
						break;
					case DTYPE_FLOAT32:
						kernel_sigmoid_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), volume(shape));
						break;
				}

				break;
			}
			case ACTIVATION_TANH:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				switch (dtype)
				{
					case DTYPE_FLOAT16:
						kernel_tanh_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), volume(shape));
						break;
					case DTYPE_FLOAT32:
						kernel_tanh_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), volume(shape));
						break;
				}

				break;
			}
			case ACTIVATION_RELU:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				switch (dtype)
				{
					case DTYPE_FLOAT16:
						kernel_relu_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), volume(shape));
						break;
					case DTYPE_FLOAT32:
						kernel_relu_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), volume(shape));
						break;
				}

				break;
			}
			case ACTIVATION_SOFTMAX:
			{
				assert(shape.rank == 2);
				const int first_dim = get_first_dim(shape);
				const int last_dim = get_last_dim(shape);

				if (last_dim == 3)
				{
					dim3 blockDim(256);
					dim3 gridDim((first_dim + 255) / 256);
					switch (dtype)
					{
						case DTYPE_FLOAT16:
							kernel_softmax_3_channels<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), first_dim);
							break;
						case DTYPE_FLOAT32:
							kernel_softmax_3_channels<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input),
									first_dim);
							break;
					}
				}
				else
				{
					dim3 blockDim(128);
					dim3 gridDim(std::min(1024, first_dim));
					switch (dtype)
					{
						case DTYPE_FLOAT16:
							kernel_softmax_generic<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), first_dim,
									last_dim);
							break;
						case DTYPE_FLOAT32:
							kernel_softmax_generic<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), first_dim,
									last_dim);
							break;
					}
				}
				break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		assert(gradient_prev != nullptr);
		assert(gradient_next != nullptr);
		assert(output != nullptr);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (gradient_prev != gradient_next)
					ml::cuda_memcpy_within_device(context, gradient_prev, 0, gradient_next, 0, sizeof(float) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				kernel_sigmoid_backward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next),
						getPointer<float>(output), volume(shape));
				break;
			}
			case ACTIVATION_TANH:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				kernel_tanh_backward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next),
						getPointer<float>(output), volume(shape));
				break;
			}
			case ACTIVATION_RELU:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				kernel_relu_backward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next),
						getPointer<float>(output), volume(shape));
				break;
			}
			case ACTIVATION_SOFTMAX:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
			mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(bias != nullptr);

		cudaStream_t stream = cuda::Context::getStream(context);
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		dim3 blockDim(128);
		dim3 gridDim(cuda::gridSize<128>(last_dim, blockDim.x), std::min(1024, first_dim));

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_add_to_last_dim<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), getPointer<half>(bias),
						first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT32:
				kernel_add_to_last_dim<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), getPointer<float>(bias),
						first_dim, last_dim, act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
