/*
 * tensor_op.cu
 *
 *  Created on: Nov 8, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
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
	__global__ void kernel_add_tensors(T *dst, float alpha1, const T *src0, float alpha2, const T *src1, int elements)
	{
		const Vector<T> a1(alpha1);
		const Vector<T> a2(alpha2);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); i < elements; i += gridDim.x * blockDim.x * vector_length<T>())
		{
			const int tmp = elements - i;
			const Vector<T> x0(src0 + i, tmp);
			const Vector<T> x1(src1 + i, tmp);
			const Vector<T> y = a1 * x0 + a2 * x1;
			y.store(dst + i, tmp);
		}
	}

	__global__ void kernel_emulate_low_precision(uint32_t *dst, const uint32_t *src, int elements)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] = src[i] & 0xFFFFF000u;
	}

	template<typename T>
	__global__ void kernel_window_partition(T *output, const T *input, int batch_size, int height, int width, int channels, int2 window_size,
			int2 offset)
	{
		const int num_windows_h = (height + window_size.x - 1) / window_size.x;
		const int num_windows_w = (width + window_size.y - 1) / window_size.y;

		const int b = blockIdx.x;
		const int h = blockIdx.y;
		const int w = blockIdx.z;

		const int x = (h + offset.x + height) % height;
		const int y = (w + offset.y + width) % width;

		const int window_idx_h = x / window_size.x;
		const int window_idx_w = y / window_size.y;

		const int idx_h = x % window_size.x;
		const int idx_w = y % window_size.y;

		const Indexer<4> input_indexer(batch_size, height, width, channels);
		const Indexer<6> output_indexer(batch_size, num_windows_h, num_windows_w, window_size.x, window_size.y, channels);

		for (int c = threadIdx.x; c < channels; c += blockDim.x)
			output[output_indexer.at(b, window_idx_h, window_idx_w, idx_h, idx_w, c)] = input[input_indexer.at(b, h, w, c)];
	}
	template<typename T>
	__global__ void kernel_window_merging(T *output, const T *input, int batch_size, int height, int width, int channels, int2 window_size,
			int2 offset)
	{
		const int num_windows_h = (height + window_size.x - 1) / window_size.x;
		const int num_windows_w = (width + window_size.y - 1) / window_size.y;

		const int b = blockIdx.x;
		const int h = blockIdx.y;
		const int w = blockIdx.z;

		const int x = (h + offset.x + height) % height;
		const int y = (w + offset.y + width) % width;

		const int window_idx_h = x / window_size.x;
		const int window_idx_w = y / window_size.y;

		const int idx_h = x % window_size.x;
		const int idx_w = y % window_size.y;

		const Indexer<6> input_indexer(batch_size, num_windows_h, num_windows_w, window_size.x, window_size.y, channels);
		const Indexer<4> output_indexer(batch_size, height, width, channels);

		for (int c = threadIdx.x; c < channels; c += blockDim.x)
			output[output_indexer.at(b, h, w, c)] = input[input_indexer.at(b, window_idx_h, window_idx_w, idx_h, idx_w, c)];
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

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src1), getPointer<half>(src2),
						length);
				break;
			case DTYPE_FLOAT32:
				kernel_multiply_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src1), getPointer<float>(src2),
						length);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, float alpha1, const void *src1, float alpha2,
			const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), alpha1, getPointer<half>(src1), alpha2,
						getPointer<half>(src2), length);
				break;
			case DTYPE_FLOAT32:
				kernel_add_tensors<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), alpha1, getPointer<float>(src1), alpha2,
						getPointer<float>(src2), length);
				break;
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

	void cuda_window_partitioning(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input,
			void *output, mlShape_t offset)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int channels = input_shape.dim[3];

		dim3 blockDim(std::min(128, channels));
		dim3 gridDim(batch_size, height, width);

		const int2 window_size { output_shape.dim[1], output_shape.dim[2] };
		const int2 window_offset { offset.dim[0], offset.dim[1] };

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT32:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT64:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<double>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_window_merging(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input, void *output,
			mlShape_t offset)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int batch_size = output_shape.dim[0];
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int channels = output_shape.dim[3];

		dim3 blockDim(std::min(128, channels));
		dim3 gridDim(batch_size, height, width);

		const int2 window_size { input_shape.dim[1], input_shape.dim[2] };
		const int2 window_offset { offset.dim[0], offset.dim[1] };

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), batch_size, height, width,
						channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT32:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT64:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<double>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */
