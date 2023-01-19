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
	__global__ void kernel_softmax_in_place_3_channels(T *output, int first_dim)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < first_dim)
		{
			float3 workspace;
			workspace.x = static_cast<float>(output[idx * 3 + 0]);
			workspace.y = static_cast<float>(output[idx * 3 + 1]);
			workspace.z = static_cast<float>(output[idx * 3 + 2]);

			const float max_value = max(workspace.x, max(workspace.y, workspace.z));
			workspace.x = exp(workspace.x - max_value);
			workspace.y = exp(workspace.y - max_value);
			workspace.z = exp(workspace.z - max_value);

			const float inv_sum = 1.0f / (workspace.x + workspace.y + workspace.z);
			workspace.x *= inv_sum;
			workspace.y *= inv_sum;
			workspace.z *= inv_sum;

			output[idx * 3 + 0] = static_cast<T>(workspace.x);
			output[idx * 3 + 1] = static_cast<T>(workspace.y);
			output[idx * 3 + 2] = static_cast<T>(workspace.z);
		}
	}

	template<typename T>
	__global__ void kernel_softmax_in_place_generic(T *output, int first_dim, int last_dim)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);
		__shared__ float workspace[1024];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 128 > tile = cg::tiled_partition<128>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float max_value = -3.4028234663e+38F; // starting with lowest possible fp32 value
			for (int j = tile.thread_rank(); j < last_dim; j += tile.size())
			{
				workspace[j] = static_cast<float>(output[i * last_dim + j]);
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
	__global__ void kernel_relu_forward_in_place(T *output, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
		{
			Vector<T> tmp(output + i, length - i);
			tmp = max(vector_zero<T>(), tmp);
			tmp.store(output + i, length - i);
		}
	}
	__global__ void kernel_relu_backward_in_place(float *gradient, const float *output, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			if (output[i] == 0.0f)
				gradient[i] = 0.0f;
	}

	template<typename T>
	__global__ void kernel_add_to_last_dim(T *input, const T *bias, int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(last_dim <= 1024);
		assert(blockDim.x == 128);

		ConstTensorWrapper<1, T> bias_wrapper(bias, last_dim);

		TensorWrapper<2, T> input_wrapper(input, first_dim, last_dim);
		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); j < last_dim; j += gridDim.x * blockDim.x * vector_length<T>())
		{
			Vector<T> _bias = bias_wrapper.load(j);
			for (int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				Vector<T> tmp = input_wrapper.load(i, j);
				tmp += _bias;
				if (act == ml::ACTIVATION_RELU)
					tmp = max(vector_zero<T>(), tmp);
				input_wrapper.store(tmp, i, j);
			}
		}
	}
}

namespace ml
{
	void cuda_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, mlActivationType_t act)
	{
		assert(input != nullptr);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				switch (dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_relu_forward_in_place<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(input), volume(shape));
						break;
					case DTYPE_FLOAT16:
						kernel_relu_forward_in_place<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(input), volume(shape));
						break;
					case DTYPE_FLOAT32:
						kernel_relu_forward_in_place<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input), volume(shape));
						break;
				}

				break;
			}
			case ACTIVATION_SOFTMAX:
			{
				int first_dim = get_first_dim(shape);
				int last_dim = get_last_dim(shape);
				if (shape.rank == 4)
				{
					if (get_last_dim(shape) > 1)
						first_dim = volume_without_last_dim(shape);
					else
						last_dim = volume_without_first_dim(shape);
				}

				if (last_dim == 3)
				{
					dim3 blockDim(256);
					dim3 gridDim((first_dim + 255) / 256);
					switch (dtype)
					{
						case DTYPE_BFLOAT16:
							kernel_softmax_in_place_3_channels<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(input), first_dim);
							break;
						case DTYPE_FLOAT16:
							kernel_softmax_in_place_3_channels<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(input), first_dim);
							break;
						case DTYPE_FLOAT32:
							kernel_softmax_in_place_3_channels<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input), first_dim);
							break;
					}
				}
				else
				{
					dim3 blockDim(128);
					dim3 gridDim(std::min(1024, first_dim));
					switch (dtype)
					{
						case DTYPE_BFLOAT16:
							kernel_softmax_in_place_generic<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(input), first_dim, last_dim);
							break;
						case DTYPE_FLOAT16:
							kernel_softmax_in_place_generic<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(input), first_dim, last_dim);
							break;
						case DTYPE_FLOAT32:
							kernel_softmax_in_place_generic<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input), first_dim, last_dim);
							break;
					}
				}
				break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output, mlActivationType_t act)
	{
		assert(gradient != nullptr);
		assert(output != nullptr);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
			{
				dim3 blockDim(256);
				dim3 gridDim = cuda::gridSize<1024>(volume(shape), 256);
				kernel_relu_backward_in_place<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient), getPointer<float>(output),
						volume(shape));
				break;
			}
			case ACTIVATION_SOFTMAX:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
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
			case DTYPE_BFLOAT16:
				kernel_add_to_last_dim<<<gridDim, blockDim, 0, stream>>>(getPointer<__nv_bfloat16 >(input), getPointer<__nv_bfloat16 >(bias),
						first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT16:
				kernel_add_to_last_dim<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(input), getPointer<half>(bias), first_dim, last_dim, act);
				break;
			case DTYPE_FLOAT32:
				kernel_add_to_last_dim<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input), getPointer<float>(bias), first_dim, last_dim, act);
				break;
		}
		if (act == ACTIVATION_SOFTMAX)
			cuda_activation_forward_in_place(context, dtype, shape, input, act);
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
