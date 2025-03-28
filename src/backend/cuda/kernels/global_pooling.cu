/*
 * global_pooling.cu
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

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

	template<typename T>
	__global__ void kernel_pooling_avg_max_forward(T *output, const T *input, int dim0, int dim1, int dim2)
	{
//		assert(blockDim.x == 32 && blockDim.y == 32);
//		__shared__ T shared_avg[32 * 32];
//		__shared__ T shared_max[32 * 32];
//
//		const int last_dim_index = (blockIdx.x * 32 + threadIdx.x) * 1;
//		const int idx = threadIdx.y * 32 + threadIdx.x;
//
//		ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);
//
//		if (last_dim_index < dim2)
//		{
//			T local_avg = get<T>(0.0f);
//			T local_max = input_wrapper.load(blockIdx.z, 0, last_dim_index);
//			for (int i = threadIdx.y; i < dim1; i += 32)
//			{
//				const T tmp = input_wrapper.load(blockIdx.z, i, last_dim_index);
//				local_avg += tmp;
//				local_max = max(local_max, tmp);
//			}
//
//			shared_avg[idx] = local_avg;
//			shared_max[idx] = local_max;
//		}
//
//		__syncthreads();
//		for (int i = 16; i >= 1; i /= 2)
//		{
//			if (threadIdx.y < i)
//			{
//				shared_avg[idx] += shared_avg[idx + i * 32];
//				shared_max[idx] = max(shared_max[idx], shared_max[idx + i * 32]);
//			}
//			__syncthreads();
//		}
//
//		if (threadIdx.y == 0 && last_dim_index < dim2)
//		{
//			TensorWrapper<3, T> output_wrapper(output, dim0, 2, dim2);
//			const T inv(1.0f / static_cast<float>(dim1));
//			const T local_avg = shared_avg[threadIdx.x] * inv;
//			const T local_max = shared_max[threadIdx.x];
//
//			output_wrapper.store(local_avg, blockIdx.z, 0, last_dim_index);
//			output_wrapper.store(local_max, blockIdx.z, 1, last_dim_index);
//		}
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

	template<typename T, int N>
	__global__ void kernel_average_pooling_forward(float beta, T *output, float alpha, const T *input, int batch_size, int hw, int channels)
	{
		assert(channels % N == 0);
		__shared__ T workspace[32][32 * N + 1];

		const int last_dim_idx = N * (32 * blockIdx.x + threadIdx.x);

		const Indexer<3> input_indexer(batch_size, hw, channels);

		vec<T, N> local_sum(0.0f);
		if (last_dim_idx < channels)
		{
			for (int i = threadIdx.y; i < hw; i += 32)
				local_sum += vec<T, N>(input + input_indexer.at(blockIdx.z, i, last_dim_idx));
		}
		for (int n = 0; n < N; n++)
			workspace[threadIdx.y][N * threadIdx.x + n] = local_sum[n];

		__syncthreads();
		for (int n = 0; n < N; n++)
			local_sum[n] = workspace[threadIdx.x][N * threadIdx.y + n];
		for (int k = 16; k >= 1; k /= 2)
			for (int n = 0; n < N; n++)
				local_sum[n] += __shfl_xor_sync(0xffffffff, local_sum[n], k);
		__syncthreads();
		if (threadIdx.x == 0)
			for (int n = 0; n < N; n++)
				workspace[0][N * threadIdx.y + n] = local_sum[n];
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < channels)
		{
			const Indexer<2> output_indexer(batch_size, channels);
			const int out_idx = output_indexer.at(blockIdx.z, last_dim_idx);
			vec<T, N> tmp;
			for (int n = 0; n < N; n++)
				tmp[n] = workspace[0][N * threadIdx.x + n];
			tmp *= vec<T, N>(alpha / static_cast<float>(hw));
			if (beta != 0.0f)
				tmp += vec<T, N>(beta) * vec<T, N>(output + out_idx);
			tmp.store(output + out_idx);
		}
	}
	template<typename T, int N>
	__global__ void kernel_average_pooling_backward(float beta, T *gradient_prev, float alpha, const T *gradient_next, int batch_size, int hw,
			int channels)
	{
		assert(channels % N == 0);
		const int hw_index = blockIdx.y * blockDim.y + threadIdx.y;
		const int channel_index = N * (blockIdx.x * blockDim.x + threadIdx.x);

		if (channel_index < channels)
		{
			const Indexer<3> input_indexer(batch_size, hw, channels);
			const Indexer<2> output_indexer(batch_size, channels);

			const vec<T, N> gradient_avg = vec<T, N>(gradient_next + output_indexer.at(blockIdx.z, channel_index)) * vec<T, N>(alpha / hw);
			for (int i = hw_index; i < hw; i += gridDim.y * blockDim.y)
			{
				const int dx_idx = input_indexer.at(blockIdx.z, i, channel_index);
				vec<T, N> tmp = gradient_avg;
				if (beta != 0.0f)
					tmp += vec<T, N>(gradient_prev + dx_idx) * vec<T, N>(beta);
				tmp.store(gradient_prev + dx_idx);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_channel_scaling_forward(float beta, T *output, float alpha, const T *input, const T *scale, int batch_size, int hw,
			int channels)
	{
		assert(channels % N == 0);
		const Indexer<2> scale_indexer(batch_size, channels);
		const Indexer<3> tensor_indexer(batch_size, hw, channels);
		const int dim0_index = blockIdx.z;
		const int dim1_index = blockIdx.y * blockDim.y + threadIdx.y;
		const int dim2_index = N * (blockIdx.x * blockDim.x + threadIdx.x);

		if (dim2_index < channels)
		{
			const vec<T, N> _scale(scale + scale_indexer.at(dim0_index, dim2_index));
			for (int i = dim1_index; i < hw; i += gridDim.y * blockDim.y)
			{
				const int idx = tensor_indexer.at(dim0_index, i, dim2_index);
				const vec<T, N> x(input + idx);
				vec<T, N> y = x * _scale * vec<T, N>(alpha);
				if (beta != 0.0f)
					y += vec<T, N>(output + idx) * vec<T, N>(beta);
				y.store(output + idx);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_channel_scaling_backward(float beta1, T *gradient_input, float beta2, T *gradient_scales, float alpha,
			const T *gradient_next, const T *input, const T *scales, int batch_size, int hw, int channels)
	{
		assert(channels % N == 0);
		assert(blockDim.x == 32 && blockDim.y == 8);
		__shared__ vec<T, N> workspace[32 * 8];

		const int global_idx = N * (blockIdx.x * blockDim.x + threadIdx.x);
		const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

		const Indexer<2> scales_indexer(batch_size, channels);

		if (global_idx < channels)
		{
			const Indexer<3> tensor_indexer(batch_size, hw, channels);

			vec<T, N> scale_gradient = vec<T, N>(0.0f);
			const vec<T, N> scale = vec<T, N>(alpha) * vec<T, N>(scales + scales_indexer.at(blockIdx.z, global_idx));
			for (int i = threadIdx.y; i < hw; i += blockDim.y)
			{
				const int tmp = tensor_indexer.at(blockIdx.z, i, global_idx);
				const vec<T, N> gradient(gradient_next + tmp);
				scale_gradient += gradient * vec<T, N>(input + tmp);

				vec<T, N> dx = scale * gradient;
				if (beta1 != 0.0f)
					dx += vec<T, N>(beta1) * vec<T, N>(gradient_input + tmp);
				dx.store(gradient_input + tmp);
			}
			workspace[local_idx] = scale_gradient;
		}

		__syncthreads();
		for (int i = blockDim.y / 2; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				workspace[local_idx] += workspace[local_idx + i * blockDim.x];
			__syncthreads();
		}

		if (threadIdx.y == 0 && global_idx < channels)
		{
			const int tmp = scales_indexer.at(blockIdx.z, global_idx);
			vec<T, N> dscales = vec<T, N>(alpha) * workspace[threadIdx.x];
			if (beta2 != 0.0f)
				dscales += vec<T, N>(beta2) * vec<T, N>(gradient_scales + tmp);
			dscales.store(gradient_scales + tmp);
		}
	}

	template<typename T>
	__global__ void kernel_global_broadcast_forward(T *output, const T *input, const T *bias, int dim0, int dim1, int dim2, mlActivationType_t act)
	{
//		ConstTensorWrapper<2, T> bias_wrapper(bias, dim0, dim2);
//
//		ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);
//		TensorWrapper<3, T> output_wrapper(output, dim0, dim1, dim2);
//		for (int j = (blockIdx.x * blockDim.x + threadIdx.x); j < dim2; j += gridDim.x * blockDim.x)
//		{
//			const T _bias = bias_wrapper.load(blockIdx.z, j);
//			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
//			{
//				T tmp = input_wrapper.load(blockIdx.z, i, j) + _bias;
//				if (act == ml::ACTIVATION_RELU)
//					tmp = ml::internal::relu(tmp);
//				if (act == ml::ACTIVATION_TANH)
//					tmp = ml::internal::tanh(tmp);
//				if (act == ml::ACTIVATION_SIGMOID)
//					tmp = ml::internal::sigmoid(tmp);
//				output_wrapper.store(tmp, blockIdx.z, i, j);
//			}
//		}
	}
	__global__ void kernel_global_broadcast_backward(float *gradient_prev, float *gradient_next, const float *output, int dim0, int dim1, int dim2,
			mlActivationType_t act)
	{
//		assert(blockDim.x == 32 && blockDim.y == 32);
//		__shared__ float workspace[32 * 32];
//
//		const int last_dim_index = blockIdx.x * 32 + threadIdx.x;
//		const int idx = threadIdx.y * 32 + threadIdx.x;
//
//		if (last_dim_index < dim2)
//		{
//			const Indexer<3> next_indexer(dim0, dim1, dim2);
//
//			float local_sum = 0.0f;
//			for (int i = threadIdx.y; i < dim1; i += 32)
//			{
//				const int index = next_indexer.at(blockIdx.z, i, last_dim_index);
//				if (act == ACTIVATION_RELU && output[index] == 0.0f)
//					gradient_next[index] = 0.0f;
//				if (act == ACTIVATION_TANH)
//					gradient_next[index] *= (1.0f - output[index]) * (1.0f + output[index]);
//				if (act == ACTIVATION_SIGMOID)
//					gradient_next[index] *= output[index] * (1.0f - output[index]);
//				local_sum += gradient_next[index];
//			}
//
//			workspace[idx] = local_sum;
//		}
//
//		__syncthreads();
//		for (int i = 16; i >= 1; i /= 2)
//		{
//			if (threadIdx.y < i)
//				workspace[idx] += workspace[idx + i * 32];
//			__syncthreads();
//		}
//
//		if (threadIdx.y == 0 && last_dim_index < dim2)
//		{
//			const Indexer<2> prev_indexer(dim0, dim2);
//			gradient_prev[prev_indexer.at(blockIdx.z, last_dim_index)] = workspace[threadIdx.x];
//		}
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

	void cuda_global_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 2);
		const int batch_size = x.dim[0];
		const int hw = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];
		assert(x.dim[0] == y.dim[0]);
		assert(x.dim[3] == y.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockDim(32, 32);
		dim3 gridDim_x4((channels + 127) / 128, 1, batch_size);
		dim3 gridDim_x1((channels + 31) / 32, 1, batch_size);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<half, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							batch_size, hw, channels);
				else
					kernel_average_pooling_forward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, hw, channels);
				else
					kernel_average_pooling_forward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_average_pooling_forward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						batch_size, hw, channels);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx)
	{
		assert(dx.rank == 4);
		assert(dy.rank == 2);
		const int batch_size = dx.dim[0];
		const int hw = dx.dim[1] * dx.dim[2];
		const int channels = dx.dim[3];
		assert(dx.dim[0] == dy.dim[0]);
		assert(dx.dim[3] == dy.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim_x4((channels + 127) / 128, (hw + 7) / 8, batch_size);
		dim3 gridDim_x1((channels + 31) / 32, (hw + 7) / 8, batch_size);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_backward<half, 4> <<<gridDim_x4,blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							batch_size, hw, channels);
				else
					kernel_average_pooling_backward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_backward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<float>(dx), alpha, data<float>(dy),
							batch_size, hw, channels);
				else
					kernel_average_pooling_backward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<float>(dx), alpha, data<float>(dy),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_average_pooling_backward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<double>(dx), alpha, data<double>(dy),
						batch_size, hw, channels);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_channel_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t scales, float beta, mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(scales.rank == 2);
		const int batch_size = x.dim[0];
		const int hw = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];
		assert(x.dim[0] == scales.dim[0]);
		assert(x.dim[3] == scales.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim_x1((channels + 31) / 32, std::max(32, (hw + 7) / 8), batch_size);
		dim3 gridDim_x4((channels + 127) / 128, std::max(32, (hw + 7) / 8), batch_size);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_channel_scaling_forward<half, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(scales), batch_size, hw, channels);
				else
					kernel_channel_scaling_forward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(scales), batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_channel_scaling_forward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(scales), batch_size, hw, channels);
				else
					kernel_channel_scaling_forward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(scales), batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT64:
				kernel_channel_scaling_forward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						data<double>(scales), batch_size, hw, channels);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_channel_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, const mlTensor_t scales,
			float beta_dx, mlTensor_t dx, float beta_scales, mlTensor_t dscales)
	{
		const int batch_size = x.dim[0];
		const int hw = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];

		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim_x4((channels + 127) / 128, 1, batch_size);
		dim3 gridDim_x1((channels + 31) / 32, 1, batch_size);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				if (channels % 4 == 0)
					kernel_channel_scaling_backward<half, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta_dx, data<half>(dx), beta_scales,
							data<half>(dscales), alpha, data<half>(dy), data<half>(x), data<half>(scales), batch_size, hw, channels);
				else
					kernel_channel_scaling_backward<half, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<half>(dx), beta_scales,
							data<half>(dscales), alpha, data<half>(dy), data<half>(x), data<half>(scales), batch_size, hw, channels);
				break;
			case DTYPE_FLOAT32:
				if (channels % 4 == 0)
					kernel_channel_scaling_backward<float, 4> <<<gridDim_x4, blockDim, 0, stream >>>(beta_dx, data<float>(dx), beta_scales,
							data<float>(dscales), alpha, data<float>(dy), data<float>(x), data<float>(scales), batch_size, hw, channels);
				else
					kernel_channel_scaling_backward<float, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<float>(dx), beta_scales,
							data<float>(dscales), alpha, data<float>(dy), data<float>(x), data<float>(scales), batch_size, hw, channels);
				break;
			case DTYPE_FLOAT64:
				kernel_channel_scaling_backward<double, 1> <<<gridDim_x1, blockDim, 0, stream >>>(beta_dx, data<double>(dx), beta_scales,
						data<double>(dscales), alpha, data<double>(dy), data<double>(x), data<double>(scales), batch_size, hw, channels);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

