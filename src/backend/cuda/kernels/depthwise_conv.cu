/*
 * depthwise_conv.cu
 *
 *  Created on: Jan 26, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../vec/vec_headers.cuh"
#include "../helpers/lines_and_tiles.cuh"
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cinttypes>
#include <iostream>

namespace
{
	using namespace vectors2;

	struct Index2D
	{
			int8_t x = 0;
			int8_t y = 0;
	};

	__device__ bool is_inside(int h, int w, int height, int width)
	{
		return 0 <= h && h < height && 0 <= w && w < width;
	}

	template<typename T>
	__device__ T get_zero();

	template<>
	__device__ float get_zero()
	{
		return 0.0f;
	}
	template<>
	__device__ half2 get_zero()
	{
		return half2(0.0f, 0.0f);
	}

	template<int KernelSize, int OutputTile, typename T>
	struct Convolution1D
	{
			__device__ Line<T, OutputTile> set(T x) const
			{
				Line<T, OutputTile> result;
				for (int i = 0; i < OutputTile; i++)
					result[i] = x;
				return result;
			}
			__device__ Line<T, KernelSize + OutputTile - 1> load_input(const T *src, int row) const
			{
				constexpr int N = KernelSize + OutputTile - 1;
				Line<T, N> result;
				const int tid = (row * N + 0) * 32 + threadIdx.x;
				for (int i = 0; i < N; i++)
					result[i] = src[tid + i * 32];
				return result;
			}
			__device__ Line<T, KernelSize> load_filter(const T *src, int row) const
			{
				Line<T, KernelSize> result;
				const int tid = (row * KernelSize + 0) * 32 + threadIdx.x;
				for (int i = 0; i < KernelSize; i++)
					result[i] = src[tid + i * 32];
				return result;
			}
			__device__ void store_output(T *dst, int row, const Line<T, OutputTile> &acc) const
			{
				const int tid = (row * OutputTile + 0) * 32 + threadIdx.x;
				for (int i = 0; i < OutputTile; i++)
					dst[tid + i * 32] = acc[i];
			}
			__device__ void accumulate(Line<T, OutputTile> &acc, const Line<T, KernelSize + OutputTile - 1> &input,
					const Line<T, KernelSize> &filter) const
			{
				for (int i = 0; i < OutputTile; i++)
				{
					T tmp = acc[i];
					for (int j = 0; j < KernelSize; j++)
						tmp += input[i + j] * filter[j];
					acc[i] = tmp;
				}
			}
	};

	template<typename T>
	struct Convolution1D<3, 4, T>
	{
			__device__ Line<T, 4> set(T x) const
			{
				Line<T, 4> result;
				result.x0 = x;
				result.x1 = x;
				result.x2 = x;
				result.x3 = x;
				return result;
			}
			__device__ Line<T, 6> load_input(const T *src, int row) const
			{
				const int tid = (row * 6 + 0) * 32 + threadIdx.x;
				Line<T, 6> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				return result;
			}
			__device__ Line<T, 3> load_filter(const T *src, int row) const
			{
				const int tid = (row * 3 + 0) * 32 + threadIdx.x;
				Line<T, 3> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				return result;
			}
			__device__ void store_output(T *dst, int row, const Line<T, 4> &acc) const
			{
				const int tid = (row * 4 + 0) * 32 + threadIdx.x;
				dst[tid + 0 * 32] = acc.x0;
				dst[tid + 1 * 32] = acc.x1;
				dst[tid + 2 * 32] = acc.x2;
				dst[tid + 3 * 32] = acc.x3;
			}
			__device__ void accumulate(Line<T, 4> &acc, const Line<T, 6> &input, const Line<T, 3> &filter) const
			{
				acc.x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2;
				acc.x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2;
				acc.x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2;
				acc.x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2;
			}
	};

	template<typename T>
	struct Convolution1D<5, 4, T>
	{
			__device__ Line<T, 4> set(T x) const
			{
				Line<T, 4> result;
				result.x0 = x;
				result.x1 = x;
				result.x2 = x;
				result.x3 = x;
				return result;
			}
			__device__ Line<T, 8> load_input(const T *src, int row) const
			{
				const int tid = (row * 8 + 0) * 32 + threadIdx.x;
				Line<T, 8> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				result.x6 = src[tid + 6 * 32];
				result.x7 = src[tid + 7 * 32];
				return result;
			}
			__device__ Line<T, 5> load_filter(const T *src, int row) const
			{
				const int tid = (row * 5 + 0) * 32 + threadIdx.x;
				Line<T, 5> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				return result;
			}
			__device__ void store_output(T *dst, int row, const Line<T, 4> &acc) const
			{
				const int tid = (row * 4 + 0) * 32 + threadIdx.x;
				dst[tid + 0 * 32] = acc.x0;
				dst[tid + 1 * 32] = acc.x1;
				dst[tid + 2 * 32] = acc.x2;
				dst[tid + 3 * 32] = acc.x3;
			}
			__device__ void accumulate(Line<T, 4> &acc, const Line<T, 8> &input, const Line<T, 5> &filter) const
			{
				acc.x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2 + input.x3 * filter.x3 + input.x4 * filter.x4;
				acc.x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2 + input.x4 * filter.x3 + input.x5 * filter.x4;
				acc.x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2 + input.x5 * filter.x3 + input.x6 * filter.x4;
				acc.x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2 + input.x6 * filter.x3 + input.x7 * filter.x4;
			}
	};

	template<typename T>
	struct Convolution1D<7, 4, T>
	{
			__device__ Line<T, 4> set(T x) const
			{
				Line<T, 4> result;
				result.x0 = x;
				result.x1 = x;
				result.x2 = x;
				result.x3 = x;
				return result;
			}
			__device__ Line<T, 10> load_input(const T *src, int row) const
			{
				const int tid = (row * 10 + 0) * 32 + threadIdx.x;
				Line<T, 10> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				result.x6 = src[tid + 6 * 32];
				result.x7 = src[tid + 7 * 32];
				result.x8 = src[tid + 8 * 32];
				result.x9 = src[tid + 9 * 32];
				return result;
			}
			__device__ Line<T, 7> load_filter(const T *src, int row) const
			{
				const int tid = (row * 7 + 0) * 32 + threadIdx.x;
				Line<T, 7> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				result.x6 = src[tid + 6 * 32];
				return result;
			}
			__device__ void store_output(T *dst, int row, const Line<T, 4> &acc) const
			{
				const int tid = (row * 4 + 0) * 32 + threadIdx.x;
				dst[tid + 0 * 32] = acc.x0;
				dst[tid + 1 * 32] = acc.x1;
				dst[tid + 2 * 32] = acc.x2;
				dst[tid + 3 * 32] = acc.x3;
			}
			__device__ void accumulate(Line<T, 4> &acc, const Line<T, 10> &input, const Line<T, 7> &filter) const
			{
				acc.x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2 + input.x3 * filter.x3 + input.x4 * filter.x4
						+ input.x5 * filter.x5 + input.x6 * filter.x6;
				acc.x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2 + input.x4 * filter.x3 + input.x5 * filter.x4
						+ input.x6 * filter.x5 + input.x7 * filter.x6;
				acc.x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2 + input.x5 * filter.x3 + input.x6 * filter.x4
						+ input.x7 * filter.x5 + input.x8 * filter.x6;
				acc.x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2 + input.x6 * filter.x3 + input.x7 * filter.x4
						+ input.x8 * filter.x5 + input.x9 * filter.x6;
			}
	};
	template<typename T>
	struct Convolution1D<4, 7, T>
	{
			__device__ Line<T, 7> set(T x) const
			{
				Line<T, 7> result;
				result.x0 = x;
				result.x1 = x;
				result.x2 = x;
				result.x3 = x;
				result.x4 = x;
				result.x5 = x;
				result.x6 = x;
				return result;
			}
			__device__ Line<T, 10> load_input(const T *src, int row) const
			{
				const int tid = (row * 10 + 0) * 32 + threadIdx.x;
				Line<T, 10> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				result.x6 = src[tid + 6 * 32];
				result.x7 = src[tid + 7 * 32];
				result.x8 = src[tid + 8 * 32];
				result.x9 = src[tid + 9 * 32];
				return result;
			}
			__device__ Line<T, 4> load_filter(const T *src, int row) const
			{
				const int tid = (row * 4 + 0) * 32 + threadIdx.x;
				Line<T, 4> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				return result;
			}
			__device__ void store_output(T *dst, int row, const Line<T, 7> &acc) const
			{
				const int tid = (row * 7 + 0) * 32 + threadIdx.x;
				dst[tid + 0 * 32] = acc.x0;
				dst[tid + 1 * 32] = acc.x1;
				dst[tid + 2 * 32] = acc.x2;
				dst[tid + 3 * 32] = acc.x3;
				dst[tid + 4 * 32] = acc.x4;
				dst[tid + 5 * 32] = acc.x5;
				dst[tid + 6 * 32] = acc.x6;
			}
			__device__ void accumulate(Line<T, 7> &acc, const Line<T, 10> &input, const Line<T, 4> &filter) const
			{
				acc.x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2 + input.x3 * filter.x3;
				acc.x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2 + input.x4 * filter.x3;
				acc.x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2 + input.x5 * filter.x3;
				acc.x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2 + input.x6 * filter.x3;
				acc.x4 += input.x4 * filter.x0 + input.x5 * filter.x1 + input.x6 * filter.x2 + input.x7 * filter.x3;
				acc.x5 += input.x5 * filter.x0 + input.x6 * filter.x1 + input.x7 * filter.x2 + input.x8 * filter.x3;
				acc.x6 += input.x6 * filter.x0 + input.x7 * filter.x1 + input.x8 * filter.x2 + input.x9 * filter.x3;
			}
	};

	template<int TileSize, int KernelSize, typename T>
	__global__ void kernel_depthwise_conv_forward(T *output, const T *input, const T *weights, const T *bias, int height, int width, int channels,
			bool invert_filter)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == TileSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ T input_tile[InputSize * InputSize * 32];
		__shared__ T filter_tile[KernelSize * KernelSize * 32];
		__shared__ T output_tile[TileSize * TileSize * 32];
		__shared__ T bias_tile[32];
		__shared__ Index2D indices[InputSize * InputSize];

		// load filters into shared memory
		const int f = blockIdx.z * blockDim.x + threadIdx.x;

		if (threadIdx.y == 0 and f < channels)
			bias_tile[threadIdx.x] = (bias == nullptr) ? get_zero<T>() : bias[f];

		for (int i = threadIdx.y; i < KernelSize * KernelSize; i += blockDim.y)
			if (f < channels)
			{
				const int tmp = invert_filter ? (KernelSize * KernelSize - 1 - i) : i;
				filter_tile[tmp * 32 + threadIdx.x] = weights[i * channels + f];
			}
		// prepare indices table
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < InputSize * InputSize; i += blockDim.x * blockDim.y)
		{
			indices[i].x = i / InputSize;
			indices[i].y = i - indices[i].x * InputSize;
		}

		__syncthreads();

		const Convolution1D<KernelSize, TileSize, T> convolution;

		for (int origin_w = 0; origin_w < width; origin_w += TileSize)
		{
			const int origin_h = TileSize * blockIdx.x;

			const Indexer<4> input_indexer(gridDim.y, height, width, channels);
			for (int i = threadIdx.y; i < InputSize * InputSize; i += blockDim.y)
			{
				const int h = origin_h + indices[i].x - Padding;
				const int w = origin_w + indices[i].y - Padding;
				if (f < channels && is_inside(h, w, height, width))
					input_tile[i * 32 + threadIdx.x] = input[input_indexer.at(blockIdx.y, h, w, f)];
				else
					input_tile[i * 32 + threadIdx.x] = get_zero<T>();
			}
			__syncthreads();

			Line<T, TileSize> acc = convolution.set(bias_tile[threadIdx.x]);
			for (int i = 0; i < KernelSize; i++)
			{
				const Line<T, InputSize> inp = convolution.load_input(input_tile, i + threadIdx.y);
				const Line<T, KernelSize> fil = convolution.load_filter(filter_tile, i);
				convolution.accumulate(acc, inp, fil);
			}
			convolution.store_output(output_tile, threadIdx.y, acc);

			for (int i = 0; i < TileSize; i++)
			{
				const int h = origin_h + threadIdx.y;
				const int w = origin_w + i;
				if (f < channels && is_inside(h, w, height, width))
					output[input_indexer.at(blockIdx.y, h, w, f)] = output_tile[(threadIdx.y * TileSize + i) * 32 + threadIdx.x];
			}
		}
	}
	template<int TileSize, int KernelSize>
	__global__ void kernel_depthwise_conv_update(const float *input, const float *gradient_next, float *weights_update, int batch_size, int height,
			int width, int channels)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == KernelSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ float input_tile[InputSize * InputSize * 32];
		__shared__ float gradient_tile[TileSize * TileSize * 32];
		__shared__ Index2D indices_in[InputSize * InputSize];
		__shared__ Index2D indices_grad[TileSize * TileSize];

		const int f = blockIdx.x * blockDim.x + threadIdx.x;

		// prepare indices table
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < InputSize * InputSize; i += blockDim.x * blockDim.y)
		{
			indices_in[i].x = i / InputSize;
			indices_in[i].y = i - indices_in[i].x * InputSize;
		}
		for (int i = tid; i < TileSize * TileSize; i += blockDim.x * blockDim.y)
		{
			indices_grad[i].x = i / TileSize;
			indices_grad[i].y = i - indices_grad[i].x * TileSize;
		}
		__syncthreads();

		const Convolution1D<TileSize, KernelSize, float> convolution;

		Line<float, KernelSize> update_acc = convolution.set(0.0f);
		for (int b = blockIdx.y; b < batch_size; b += gridDim.y)
			for (int origin_h = 0; origin_h < height; origin_h += TileSize)
				for (int origin_w = 0; origin_w < width; origin_w += TileSize)
				{
					const Indexer<4> input_indexer(batch_size, height, width, channels);
					for (int i = threadIdx.y; i < InputSize * InputSize; i += blockDim.y)
					{
						const int h = origin_h + indices_in[i].x - Padding;
						const int w = origin_w + indices_in[i].y - Padding;
						if (f < channels && is_inside(h, w, height, width))
							input_tile[i * 32 + threadIdx.x] = input[input_indexer.at(b, h, w, f)];
						else
							input_tile[i * 32 + threadIdx.x] = 0.0f;
					}
					for (int i = threadIdx.y; i < TileSize * TileSize; i += blockDim.y)
					{
						const int h = origin_h + indices_grad[i].x;
						const int w = origin_w + indices_grad[i].y;
						if (f < channels && is_inside(h, w, height, width))
							gradient_tile[i * 32 + threadIdx.x] = gradient_next[input_indexer.at(b, h, w, f)];
						else
							gradient_tile[i * 32 + threadIdx.x] = 0.0f;
					}
					__syncthreads();

					for (int i = 0; i < TileSize; i++)
					{
						const Line<float, InputSize> inp = convolution.load_input(input_tile, i + threadIdx.y);
						const Line<float, TileSize> fil = convolution.load_filter(gradient_tile, i);
						convolution.accumulate(update_acc, inp, fil);
					}
					__syncthreads();
				}

		convolution.store_output(input_tile, threadIdx.y, update_acc); // reusing input tile to save shared memory
		__syncthreads();
		if (f < channels)
		{
			const Indexer<3> update_indexer(gridDim.y, KernelSize * KernelSize, channels);
			for (int i = threadIdx.y; i < KernelSize * KernelSize; i += blockDim.y)
				weights_update[update_indexer.at(blockIdx.y, i, f)] = input_tile[i * 32 + threadIdx.x];
		}
	}
	__global__ void kernel_sum_update(float *dst, const float *src, int first_dim, int last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		assert(gridDim.y == 1);
		__shared__ float workspace[32][32 + 1];

		const int last_dim_idx = 32 * blockIdx.x + threadIdx.x;
		float local_sum = 0.0f;
		if (last_dim_idx < last_dim)
		{
			for (int i = threadIdx.y; i < first_dim; i += 32)
			{
				const float tmp = src[i * last_dim + last_dim_idx];
				local_sum += tmp;
			}
			workspace[threadIdx.y][threadIdx.x + 0] = local_sum;
		}
		__syncthreads();
		local_sum = workspace[threadIdx.x][threadIdx.y];
		for (int k = 16; k >= 1; k /= 2)
			local_sum += __shfl_xor_sync(0xffffffff, local_sum, k);
		__syncthreads();
		if (threadIdx.x == 0)
			workspace[0][threadIdx.y] = local_sum;
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
			dst[last_dim_idx] = workspace[0][threadIdx.x];
	}
}

namespace ml
{
	void cuda_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, const void *bias, void *output)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);

		const int tmp = (dtype == DTYPE_FLOAT32) ? 32 : 64;
		constexpr int TileSize = 4;

		const int num_tiles_h = (height + TileSize - 1) / TileSize;

		dim3 blockDim(32, TileSize);
		dim3 gridDim(num_tiles_h, batch_size, (channels + tmp - 1) / tmp);

		if (dtype == DTYPE_FLOAT32)
		{
			switch (filter_size)
			{
				case 3:
					kernel_depthwise_conv_forward<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input),
							getPointer<float>(weights), getPointer<float>(bias), height, width, channels, false);
					break;
				case 5:
					kernel_depthwise_conv_forward<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input),
							getPointer<float>(weights), getPointer<float>(bias), height, width, channels, false);
					break;
				case 7:
					kernel_depthwise_conv_forward<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input),
							getPointer<float>(weights), getPointer<float>(bias), height, width, channels, false);
					break;
				default:
					break;
			}
		}
		if (dtype == DTYPE_FLOAT16)
		{
			assert(channels % 2 == 0);
			switch (filter_size)
			{
				case 3:
					kernel_depthwise_conv_forward<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(getPointer<half2>(output), getPointer<half2>(input),
							getPointer<half2>(weights), getPointer<half2>(bias), height, width, channels / 2, false);
					break;
				case 5:
					kernel_depthwise_conv_forward<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(getPointer<half2>(output), getPointer<half2>(input),
							getPointer<half2>(weights), getPointer<half2>(bias), height, width, channels / 2, false);
					break;
				case 7:
					kernel_depthwise_conv_forward<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(getPointer<half2>(output), getPointer<half2>(input),
							getPointer<half2>(weights), getPointer<half2>(bias), height, width, channels / 2, false);
					break;
				default:
					break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_depthwise_conv_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *gradient_next,
			const void *weights, void *gradient_prev)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);

		constexpr int TileSize = 4;

		const int num_tiles_h = (height + TileSize - 1) / TileSize;

		dim3 blockDim(32, TileSize);
		dim3 gridDim(num_tiles_h, batch_size, (channels + 31) / 32);

		switch (filter_size)
		{
			case 3:
				kernel_depthwise_conv_forward<TileSize, 3, float> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev),
						getPointer<float>(gradient_next), getPointer<float>(weights), nullptr, height, width, channels, true);
				break;
			case 5:
				kernel_depthwise_conv_forward<TileSize, 5, float> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev),
						getPointer<float>(gradient_next), getPointer<float>(weights), nullptr, height, width, channels, true);
				break;
			case 7:
				kernel_depthwise_conv_forward<TileSize, 7, float> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradient_prev),
						getPointer<float>(gradient_next), getPointer<float>(weights), nullptr, height, width, channels, true);
				break;
			default:
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_depthwise_conv_update(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input, const void *gradient_next,
			void *weights_update)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudaStream_t stream = cuda::Context::getStream(context);
		float *workspace = getPointer<float>(cuda::Context::getWorkspace(context));
		const int workspace_size = cuda::Context::getWorkspaceSize(context);

		const int last_dim = filter_size * filter_size * channels;
		const int max_blocks = workspace_size / (sizeof(float) * last_dim);

		constexpr int TileSize = 4;

		const int num_blocks = std::min(1024, std::min(batch_size, max_blocks));

		dim3 blockDim(32, filter_size);
		dim3 gridDim((channels + 31) / 32, num_blocks);

		switch (filter_size)
		{
			case 3:
				kernel_depthwise_conv_update<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input),
						getPointer<float>(gradient_next), getPointer<float>(workspace), batch_size, height, width, channels);
				break;
			case 5:
				kernel_depthwise_conv_update<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input),
						getPointer<float>(gradient_next), getPointer<float>(workspace), batch_size, height, width, channels);
				break;
			case 7:
				kernel_depthwise_conv_update<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(input),
						getPointer<float>(gradient_next), getPointer<float>(workspace), batch_size, height, width, channels);
				break;
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		blockDim = dim3(32, 32);
		gridDim = dim3((last_dim + 31) / 32);
		kernel_sum_update<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(weights_update), getPointer<float>(workspace), num_blocks, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

