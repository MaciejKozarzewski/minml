/*
 * global_pooling.cu
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/tensor_wrappers.cuh"
#include "../vectors/vectors.cuh"

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

	template<typename T>
	__global__ void kernel_pooling_avg_max_forward(T *output, const T *input, int dim0, int dim1, int dim2)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ Vector<T> shared_avg[32 * 32];
		__shared__ Vector<T> shared_max[32 * 32];

		const int last_dim_index = (blockIdx.x * 32 + threadIdx.x) * vector_length<T>();
		const int idx = threadIdx.y * 32 + threadIdx.x;

		ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);

		if (last_dim_index < dim2)
		{
			Vector<T> local_avg = vector_zero<T>();
			Vector<T> local_max = input_wrapper.load(blockIdx.z, 0, last_dim_index);
			for (int i = threadIdx.y; i < dim1; i += 32)
			{
				const Vector<T> tmp = input_wrapper.load(blockIdx.z, i, last_dim_index);
				local_avg += tmp;
				local_max = max(local_max, tmp);
			}

			shared_avg[idx] = local_avg;
			shared_max[idx] = local_max;
		}

		__syncthreads();
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
			{
				shared_avg[idx] += shared_avg[idx + i * 32];
				shared_max[idx] = max(shared_max[idx], shared_max[idx + i * 32]);
			}
			__syncthreads();
		}

		if (threadIdx.y == 0 && last_dim_index < dim2)
		{
			TensorWrapper<3, T> output_wrapper(output, dim0, 2, dim2);
			const Vector<T> inv(1.0f / static_cast<float>(dim1));
			const Vector<T> local_avg = shared_avg[threadIdx.x] * inv;
			const Vector<T> local_max = shared_max[threadIdx.x];

			output_wrapper.store(local_avg, blockIdx.z, 0, last_dim_index);
			output_wrapper.store(local_max, blockIdx.z, 1, last_dim_index);
		}
	}
	__global__ void kernel_pooling_avg_max_backward(float *gradient_prev, const float *gradient_next, const float *input, const float *output,
			int dim0, int dim1, int dim2)
	{
		const int last_dim_index = blockIdx.x * blockDim.x + threadIdx.x;

		if (last_dim_index < dim2)
		{
			const Indexer<3> input_indexer(dim0, dim1, dim2);
			const Indexer<3> output_indexer(dim0, 2, dim2);

			const float gradient_avg = gradient_next[output_indexer.at(blockIdx.z, 0, last_dim_index)] / static_cast<float>(dim1);
			const float gradient_max = gradient_next[output_indexer.at(blockIdx.z, 1, last_dim_index)];
			const float local_max = output[output_indexer.at(blockIdx.z, 1, last_dim_index)];

			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
			{
				const int index = input_indexer.at(blockIdx.z, i, last_dim_index);
				const float d_max = (input[index] == local_max) ? gradient_max : 0.0f;
				gradient_prev[index] = gradient_avg + d_max;
			}
		}
	}

	template<typename T, typename U>
	__global__ void kernel_average_pooling_forward(U *output, const T *input, int dim0, int dim1, int dim2, float scale, float shift)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ float shared_avg[32 * 32];

		const int last_dim_index = blockIdx.x * 32 + threadIdx.x;
		const int idx = threadIdx.y * 32 + threadIdx.x;

		const Indexer<3> input_indexer(dim0, dim1, dim2);

		if (last_dim_index < dim2)
		{
			float local_avg = 0.0f;
			for (int i = threadIdx.y; i < dim1; i += 32)
				local_avg += static_cast<float>(input[input_indexer.at(blockIdx.z, i, last_dim_index)]) * scale + shift;
			shared_avg[idx] = local_avg;
		}

		__syncthreads();
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				shared_avg[idx] += shared_avg[idx + i * 32];
			__syncthreads();
		}

		if (threadIdx.y == 0 && last_dim_index < dim2)
		{
			const Indexer<2> output_indexer(dim0, dim2);
			output[output_indexer.at(blockIdx.z, last_dim_index)] = shared_avg[threadIdx.x] / static_cast<float>(dim1);
		}
	}
	__global__ void kernel_average_pooling_backward(float *gradient_prev, const float *gradient_next, int dim0, int dim1, int dim2)
	{
		const int last_dim_index = blockIdx.x * blockDim.x + threadIdx.x;

		if (last_dim_index < dim2)
		{
			const Indexer<3> input_indexer(dim0, dim1, dim2);
			const Indexer<2> output_indexer(dim0, dim2);

			const float gradient_avg = gradient_next[output_indexer.at(blockIdx.z, last_dim_index)] / static_cast<float>(dim1);

			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
				gradient_prev[input_indexer.at(blockIdx.z, i, last_dim_index)] = gradient_avg;
		}
	}
	template<typename T, typename U>
	__global__ void kernel_channel_scaling_forward(U *output, const U *input, const T *scale, int dim0, int dim1, int dim2)
	{
		const Indexer<2> scale_indexer(dim0, dim2);
		const Indexer<3> tensor_indexer(dim0, dim1, dim2);

		for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < dim2; j += gridDim.x * blockDim.x)
		{
			const T _scale = scale[scale_indexer.at(blockIdx.z, j)];
			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
			{
				const int idx = tensor_indexer.at(blockIdx.z, i, j);
				output[idx] = input[idx] * _scale;
			}
		}
	}
	__global__ void kernel_channel_scaling_backward(float *gradient_input, float *gradient_scales, const float *gradient_next, const float *input,
			const float *scales, int dim0, int dim1, int dim2)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ float workspace[32 * 32];

		const int last_dim_index = blockIdx.x * 32 + threadIdx.x;
		const int idx = threadIdx.y * 32 + threadIdx.x;

		const Indexer<2> scales_indexer(dim0, dim2);

		if (last_dim_index < dim2)
		{
			const Indexer<3> tensor_indexer(dim0, dim1, dim2);

			float scale_gradient = 0.0f;
			for (int i = threadIdx.y; i < dim1; i += 32)
			{
				const int tmp = tensor_indexer.at(blockIdx.z, i, last_dim_index);
				const float scale = scales[scales_indexer.at(blockIdx.z, last_dim_index)];
				const float gradient = gradient_next[tmp];
				scale_gradient += gradient * input[tmp];
				gradient_input[tmp] = scale * gradient;
			}
			workspace[idx] = scale_gradient;
		}

		__syncthreads();
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				workspace[idx] += workspace[idx + i * 32];
			__syncthreads();
		}

		if (threadIdx.y == 0 && last_dim_index < dim2)
			gradient_scales[scales_indexer.at(blockIdx.z, last_dim_index)] = workspace[threadIdx.x];
	}

	template<typename T>
	__global__ void kernel_global_broadcast_forward(T *output, const T *input, const T *bias, int dim0, int dim1, int dim2, mlActivationType_t act)
	{
		ConstTensorWrapper<2, T> bias_wrapper(bias, dim0, dim2);

		ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);
		TensorWrapper<3, T> output_wrapper(output, dim0, dim1, dim2);
		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); j < dim2; j += gridDim.x * blockDim.x * vector_length<T>())
		{
			const Vector<T> _bias = bias_wrapper.load(blockIdx.z, j);
			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
			{
				Vector<T> tmp = input_wrapper.load(blockIdx.z, i, j) + _bias;
				if (act == ml::ACTIVATION_RELU)
					tmp = vectors::max(vector_zero<T>(), tmp);
				if (act == ml::ACTIVATION_TANH)
					tmp = vectors::tanh(tmp);
				if (act == ml::ACTIVATION_SIGMOID)
					tmp = vector_one<T>() / (vector_one<T>() + vectors::exp(-tmp));
				output_wrapper.store(tmp, blockIdx.z, i, j);
			}
		}
	}
	__global__ void kernel_global_broadcast_backward(float *gradient_prev, float *gradient_next, const float *output, int dim0, int dim1, int dim2,
			mlActivationType_t act)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ float workspace[32 * 32];

		const int last_dim_index = blockIdx.x * 32 + threadIdx.x;
		const int idx = threadIdx.y * 32 + threadIdx.x;

		if (last_dim_index < dim2)
		{
			const Indexer<3> next_indexer(dim0, dim1, dim2);

			float local_sum = 0.0f;
			for (int i = threadIdx.y; i < dim1; i += 32)
			{
				const int index = next_indexer.at(blockIdx.z, i, last_dim_index);
				if (act == ACTIVATION_RELU && output[index] == 0.0f)
					gradient_next[index] = 0.0f;
				if (act == ACTIVATION_TANH)
					gradient_next[index] *= (1.0f - output[index]) * (1.0f + output[index]);
				if (act == ACTIVATION_SIGMOID)
					gradient_next[index] *= output[index] * (1.0f - output[index]);
				local_sum += gradient_next[index];
			}

			workspace[idx] = local_sum;
		}

		__syncthreads();
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				workspace[idx] += workspace[idx + i * 32];
			__syncthreads();
		}

		if (threadIdx.y == 0 && last_dim_index < dim2)
		{
			const Indexer<2> prev_indexer(dim0, dim2);
			gradient_prev[prev_indexer.at(blockIdx.z, last_dim_index)] = workspace[threadIdx.x];
		}
	}
}

namespace ml
{
	void cuda_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		dim3 blockDim(32, 32);
		dim3 gridDim((dim2 + 31) / 32, 1, dim0);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_pooling_avg_max_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<half>(output), getPointer<half>(input), dim0, dim1,
						dim2);
				break;
			case DTYPE_FLOAT32:
				kernel_pooling_avg_max_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(output), getPointer<float>(input), dim0, dim1,
						dim2);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input, const void *output)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 gridDim((dim2 + 127) / 128, std::max(256, dim1), dim0);

		kernel_pooling_avg_max_backward<<<gridDim, 128, 0, stream >>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next),
				getPointer<float>(input), getPointer<float>(output), dim0, dim1, dim2);

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
			mlActivationType_t act)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 gridDim((dim2 + 127) / 128, std::max(256, dim1), dim0);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_global_broadcast_forward<<<gridDim, 128, 0, stream >>>(getPointer<half>(output), getPointer<half>(input),
						getPointer<half>(bias), dim0, dim1, dim2, act);
				break;
			case DTYPE_FLOAT32:
				kernel_global_broadcast_forward<<<gridDim, 128, 0, stream >>>(getPointer<float>(output), getPointer<float>(input),
						getPointer<float>(bias), dim0, dim1, dim2, act);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		dim3 blockDim(32, 32);
		dim3 gridDim((dim2 + 31) / 32, 1, dim0);

		cudaStream_t stream = cuda::Context::getStream(context);
		kernel_global_broadcast_backward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next),
				getPointer<float>(output), dim0, dim1, dim2, act);
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_global_average_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		dim3 blockDim(32, 32);
		dim3 gridDim((dim2 + 31) / 32, 1, dim0);

		const float scale = 1.0f;
		const float shift = 0.0f;

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_average_pooling_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<half>(output), getPointer<half>(input), dim0, dim1, dim2,
						scale, shift);
				break;
			case DTYPE_FLOAT32:
				kernel_average_pooling_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(output), getPointer<float>(input), dim0, dim1,
						dim2, scale, shift);
				break;
			case DTYPE_FLOAT64:
				kernel_average_pooling_forward<<<gridDim, blockDim, 0, stream >>>(getPointer<double>(output), getPointer<double>(input), dim0, dim1,
						dim2, scale, shift);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_average_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 gridDim((dim2 + 127) / 128, std::max(256, dim1), dim0);

		kernel_average_pooling_backward<<<gridDim, 128, 0, stream >>>(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), dim0, dim1,
				dim2);

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_channel_scaling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *scales)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 gridDim((dim2 + 127) / 128, std::max(256, dim1), dim0);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_channel_scaling_forward<<<gridDim, 128, 0, stream >>>(getPointer<half>(output), getPointer<half>(input),
						getPointer<half>(scales), dim0, dim1, dim2);
				break;
			case DTYPE_FLOAT32:
				kernel_channel_scaling_forward<<<gridDim, 128, 0, stream >>>(getPointer<float>(output), getPointer<float>(input),
						getPointer<float>(scales), dim0, dim1, dim2);
				break;
			case DTYPE_FLOAT64:
				kernel_channel_scaling_forward<<<gridDim, 128, 0, stream >>>(getPointer<double>(output), getPointer<double>(input),
						getPointer<double>(scales), dim0, dim1, dim2);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_channel_scaling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev_0, void *gradient_prev_1, const void *gradient_next,
			const void *input_0, const void *input_1)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		dim3 blockDim(32, 32);
		dim3 gridDim((dim2 + 31) / 32, 1, dim0);

		cudaStream_t stream = cuda::Context::getStream(context);
		kernel_channel_scaling_backward<<<gridDim, blockDim, 0, stream >>>(getPointer<float>(gradient_prev_0), getPointer<float>(gradient_prev_1),
				getPointer<float>(gradient_next), getPointer<float>(input_0), getPointer<float>(input_1), dim0, dim1, dim2);
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

