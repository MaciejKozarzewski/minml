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

	template<typename T, int N, typename U>
	__global__ void kernel_average_pooling_forward(float beta, T *output, float alpha, const T *input, int batch_size, int hw, int channels)
	{
		assert(channels % N == 0);
		__shared__ U workspace[32][32 * N + 1];
		if (blockIdx.z != 16)
			return;

		const int last_dim_idx = N * (32 * blockIdx.x + threadIdx.x);

		const Indexer<3> input_indexer(batch_size, hw, channels);

		vec<U, N> local_sum(0.0f);
		if (last_dim_idx < channels)
		{
			for (int i = threadIdx.y; i < hw; i += 32)
				local_sum += load_vec<U, N>(input + input_indexer.at(blockIdx.z, i, last_dim_idx));
		}
		for (int n = 0; n < N; n++)
			workspace[threadIdx.y][N * threadIdx.x + n] = local_sum[n];

		__syncthreads();
		vec<U, N> reduction_sum;
		for (int n = 0; n < N; n++)
			reduction_sum[n] = workspace[threadIdx.x][N * threadIdx.y + n];
		for (int k = 16; k >= 1; k /= 2)
			for (int n = 0; n < N; n++)
				reduction_sum[n] += __shfl_xor_sync(0xffffffff, reduction_sum[n], k);
		__syncthreads();
		if (threadIdx.x == 0)
			for (int n = 0; n < N; n++)
				workspace[0][N * threadIdx.y + n] = reduction_sum[n];
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < channels)
		{
			const Indexer<2> output_indexer(batch_size, channels);
			const int out_idx = output_indexer.at(blockIdx.z, last_dim_idx);
			vec<U, N> tmp;
			for (int n = 0; n < N; n++)
				tmp[n] = workspace[0][N * threadIdx.x + n];
			tmp *= vec<U, N>(alpha / static_cast<float>(hw));
			if (beta != 0.0f)
				tmp += vec<U, N>(beta) * load_vec<U, N>(output + out_idx);
			store_vec(output + out_idx, tmp);
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
	__global__ void kernel_channel_scaling_forward(float beta, T *output, float alpha, const T *input, const T *scales, int batch_size, int hw,
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
			const vec<T, N> scale(scales + scale_indexer.at(dim0_index, dim2_index));
			for (int i = dim1_index; i < hw; i += gridDim.y * blockDim.y)
			{
				const int idx = tensor_indexer.at(dim0_index, i, dim2_index);
				const vec<T, N> x(input + idx);
				vec<T, N> y = x * scale * vec<T, N>(alpha);
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

	template<typename T, int N>
	__global__ void kernel_channel_average_pooling_forward(float beta, T *output, float alpha, const T *input, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
		{
			vec<T, N> local_sum(0.0f);
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
				local_sum += vec<T, N>(input + i * last_dim + j);

			float avg = horizontal_add(local_sum);
			for (int k = 16; k >= 1; k /= 2)
				avg += __shfl_xor_sync(0xffffffff, avg, k);
			if (threadIdx.x == 0)
			{
				avg *= alpha / last_dim;
				if (beta != 0.0f)
					avg += beta * static_cast<float>(output[i]);
				output[i] = avg;
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_channel_average_pooling_backward(float beta, T *gradient_prev, float alpha, const T *gradient_next, int first_dim,
			int last_dim)
	{
		assert(last_dim % N == 0);
		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
		{
			const float gradient_avg = static_cast<float>(gradient_next[i]) * alpha / last_dim;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				vec<T, N> tmp(gradient_avg);
				if (beta != 0.0f)
					tmp += vec<T, N>(gradient_prev + i * last_dim + j) * vec<T, N>(beta);
				tmp.store(gradient_prev + i * last_dim + j);
			}
		}
	}

	template<typename T, int N>
	__global__ void kernel_spatial_scaling_forward(float beta, T *output, float alpha, const T *input, const T *scales, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
		{
			const vec<T, N> scale(static_cast<float>(scales[i]) * alpha);
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> x(input + i * last_dim + j);
				vec<T, N> y = x * scale;
				if (beta != 0.0f)
					y += vec<T, N>(output + i * last_dim + j) * vec<T, N>(beta);
				y.store(output + i * last_dim + j);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_spatial_scaling_backward(float beta1, T *gradient_input, float beta2, T *gradient_scales, float alpha,
			const T *gradient_next, const T *input, const T *scales, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
		{
			vec<T, N> scale_gradient(0.0f);
			const vec<T, N> scale(static_cast<float>(scales[i]) * alpha);
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> dy(gradient_next + i * last_dim + j);

				scale_gradient += dy * vec<T, N>(input + i * last_dim + j);

				vec<T, N> dx = dy * scale;
				if (beta1 != 0.0f)
					dx += vec<T, N>(gradient_input + i * last_dim + j) * vec<T, N>(beta1);
				dx.store(gradient_input + i * last_dim + j);
			}

			float dscale = horizontal_add(scale_gradient);
			for (int k = 16; k >= 1; k /= 2)
				dscale += __shfl_xor_sync(0xffffffff, dscale, k);
			if (threadIdx.x == 0)
			{
				dscale *= alpha;
				if (beta2 != 0.0f)
					dscale += beta2 * static_cast<float>(gradient_scales[i]);
				gradient_scales[i] = dscale;
			}
		}
	}

}

namespace ml
{
	void cuda_global_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 2);
		const int batch_size = x.dim[0];
		const int hw = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];
		assert(x.dim[0] == y.dim[0]);
		assert(x.dim[3] == y.dim[1]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 32);
		dim3 gridDim_x4((channels + 127) / 128, 1, batch_size);
		dim3 gridDim_x1((channels + 31) / 32, 1, batch_size);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<half, 4, float> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							batch_size, hw, channels);
				else
					kernel_average_pooling_forward<half, 1, float> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_average_pooling_forward<float, 4, float> <<<gridDim_x4, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, hw, channels);
				else
					kernel_average_pooling_forward<float, 1, float> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							batch_size, hw, channels);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_average_pooling_forward<double, 1, double> <<<gridDim_x1, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
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

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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

	void cuda_channel_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 2);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(y.dim[0] == x.dim[0]);
		assert(y.dim[1] == x.dim[1] * x.dim[2]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim(1, std::min(2048, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_channel_average_pooling_forward<half, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							first_dim, last_dim);
				else
					kernel_channel_average_pooling_forward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_channel_average_pooling_forward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							first_dim, last_dim);
				else
					kernel_channel_average_pooling_forward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_channel_average_pooling_forward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						first_dim, last_dim);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_channel_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx)
	{
		assert(dx.rank == 4);
		assert(dy.rank == 2);
		const int first_dim = volume_without_last_dim(dx);
		const int last_dim = get_last_dim(dx);
		assert(dy.dim[0] == dx.dim[0]);
		assert(dy.dim[1] == dx.dim[1] * dx.dim[2]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim(1, std::min(2048, (first_dim + 7) / 8));
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_channel_average_pooling_backward<half, 4> <<<gridDim,blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							first_dim, last_dim);
				else
					kernel_channel_average_pooling_backward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(dx), alpha, data<half>(dy),
							first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_channel_average_pooling_backward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(dx), alpha,
							data<float>(dy), first_dim, last_dim);
				else
					kernel_channel_average_pooling_backward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(dx), alpha,
							data<float>(dy), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_channel_average_pooling_backward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<double>(dx), alpha,
						data<double>(dy), first_dim, last_dim);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_spatial_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t scales, float beta, mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(scales.rank == 2);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(scales.dim[0] == x.dim[0]);
		assert(scales.dim[1] == x.dim[1] * x.dim[2]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim(1, std::min(2048, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_spatial_scaling_forward<half, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(scales), first_dim, last_dim);
				else
					kernel_spatial_scaling_forward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<half>(y), alpha, data<half>(x),
							data<half>(scales), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_spatial_scaling_forward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(scales), first_dim, last_dim);
				else
					kernel_spatial_scaling_forward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<float>(y), alpha, data<float>(x),
							data<float>(scales), first_dim, last_dim);
				break;
			}
			case DTYPE_FLOAT64:
				kernel_spatial_scaling_forward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta, data<double>(y), alpha, data<double>(x),
						data<double>(scales), first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_spatial_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, const mlTensor_t scales,
			float beta_dx, mlTensor_t dx, float beta_scales, mlTensor_t dscales)
	{
		assert(dx.rank == 4);
		assert(dy.rank == 4);
		assert(dscales.rank == 2);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(dscales.dim[0] == dx.dim[0]);
		assert(dscales.dim[1] == dx.dim[1] * dx.dim[2]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim(1, std::min(2048, (first_dim + 7) / 8));
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				if (last_dim % 4 == 0)
					kernel_spatial_scaling_backward<half, 4> <<<gridDim, blockDim, 0, stream >>>(beta_dx, data<half>(dx), beta_scales,
							data<half>(dscales), alpha, data<half>(dy), data<half>(x), data<half>(scales), first_dim, last_dim);
				else
					kernel_spatial_scaling_backward<half, 1> <<<gridDim, blockDim, 0, stream >>>(beta_dx, data<half>(dx), beta_scales,
							data<half>(dscales), alpha, data<half>(dy), data<half>(x), data<half>(scales), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				if (last_dim % 4 == 0)
					kernel_spatial_scaling_backward<float, 4> <<<gridDim, blockDim, 0, stream >>>(beta_dx, data<float>(dx), beta_scales,
							data<float>(dscales), alpha, data<float>(dy), data<float>(x), data<float>(scales), first_dim, last_dim);
				else
					kernel_spatial_scaling_backward<float, 1> <<<gridDim, blockDim, 0, stream >>>(beta_dx, data<float>(dx), beta_scales,
							data<float>(dscales), alpha, data<float>(dy), data<float>(x), data<float>(scales), first_dim, last_dim);
				break;
			case DTYPE_FLOAT64:
				kernel_spatial_scaling_backward<double, 1> <<<gridDim, blockDim, 0, stream >>>(beta_dx, data<double>(dx), beta_scales,
						data<double>(dscales), alpha, data<double>(dy), data<double>(x), data<double>(scales), first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

