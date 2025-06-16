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
#include "../helpers/misc.cuh"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cinttypes>
#include <iostream>

namespace
{
	using namespace vectors;

	__device__ float to_fp32(const vec1f x)
	{
		return x.x0;
	}
	__device__ float to_fp32(const vec1h x)
	{
		return static_cast<float>(x.x0);
	}
	template<typename T>
	__device__ T to_vec(float x)
	{
		return T{};
	}

	template<>
	__device__ vec1f to_vec<vec1f>(float x)
	{
		return vec1f(x);
	}
	template<>
	__device__ vec1h to_vec<vec1h>(float x)
	{
		return vec1h(x);
	}
	template<>
	__device__ vec2h to_vec<vec2h>(float x)
	{
		return vec2h(x);
	}

	struct Index2D
	{
			int8_t x = 0;
			int8_t y = 0;
	};

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

	template<int TileSize, int KernelSize, typename T>
	__global__ void kernel_depthwise_conv_forward(float beta, T *y_ptr, float alpha, const T *x_ptr, const T *w_ptr, const T *b_ptr, int height,
			int width, int channels, bool invert_filter)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == TileSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ T filter_tile[KernelSize * KernelSize * 32];
		__shared__ T bias_tile[32];

		const int f = blockIdx.z * blockDim.x + threadIdx.x;

		if (f < channels)
		{
			if (threadIdx.y == 0)
				bias_tile[threadIdx.x] = (b_ptr == nullptr) ? to_vec<T>(0.0f) : b_ptr[f];
			for (int i = threadIdx.y; i < KernelSize * KernelSize; i += blockDim.y)
			{
				const int tmp = invert_filter ? (KernelSize * KernelSize - 1 - i) : i;
				filter_tile[tmp * 32 + threadIdx.x] = w_ptr[i * channels + f];
			}
		}
		__syncthreads();

		if (f < channels)
		{
			const Convolution1D<KernelSize, TileSize, T> convolution;
			const T bias = bias_tile[threadIdx.x];

			const int h_stride = channels * width;
			const int w_stride = channels;

			for (int origin_w = 0; origin_w < width; origin_w += TileSize)
			{
				const int origin_h = TileSize * blockIdx.x;
				const int input_offset = Indexer<4>(gridDim.y, height, width, channels).at(blockIdx.y, 0, 0, f);
				Line<T, TileSize> acc = convolution.set(to_vec<T>(0.0f));
				for (int k = 0; k < KernelSize; k++)
				{
					const int h = origin_h + threadIdx.y + k - Padding;

					if (is_inside(h, height))
					{
						Line<T, InputSize> inp;
						for (int i = 0; i < InputSize; i++)
						{
							const int w = origin_w + i - Padding;
							inp[i] = is_inside(w, width) ? x_ptr[input_offset + h * h_stride + w * w_stride] : to_vec<T>(0.0f);
						}

						const Line<T, KernelSize> fil = convolution.load_filter(filter_tile, k);
						convolution.accumulate(acc, inp, fil);
					}
				}

				for (int i = 0; i < TileSize; i++)
				{
					const int h = origin_h + threadIdx.y;
					const int w = origin_w + i;
					if (is_inside(h, w, height, width))
					{
						const int idx = input_offset + h * h_stride + w * w_stride;
						T tmp = acc[i] * to_vec<T>(alpha) + bias;
						if (beta != 0.0f)
							tmp += to_vec<T>(beta) * y_ptr[idx];
						y_ptr[idx] = tmp;
					}
				}
			}
		}
	}
	template<int TileSize, int KernelSize, typename T, typename U>
	__global__ void kernel_depthwise_conv_update(T *dw_ptr, const U *x_ptr, const U *dy_ptr, int batch_size, int height, int width, int channels)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == KernelSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ T gradient_tile[TileSize * TileSize * 32];
		__shared__ Index2D indices_grad[TileSize * TileSize];

		const int f = blockIdx.x * blockDim.x + threadIdx.x;

		// prepare indices table
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < TileSize * TileSize; i += blockDim.x * blockDim.y)
		{
			indices_grad[i].x = i / TileSize;
			indices_grad[i].y = i - indices_grad[i].x * TileSize;
		}
		__syncthreads();

		const Convolution1D<TileSize, KernelSize, T> convolution;

		if (f < channels)
		{
			Line<T, KernelSize> update_acc = convolution.set(to_vec<T>(0.0f));
			for (int b = blockIdx.y; b < batch_size; b += gridDim.y)
				for (int origin_h = 0; origin_h < height; origin_h += TileSize)
					for (int origin_w = 0; origin_w < width; origin_w += TileSize)
					{
						const Indexer<4> input_indexer(batch_size, height, width, channels);
						for (int i = threadIdx.y; i < TileSize * TileSize; i += blockDim.y)
						{
							const int h = origin_h + indices_grad[i].x;
							const int w = origin_w + indices_grad[i].y;
							if (is_inside(h, w, height, width))
								gradient_tile[i * 32 + threadIdx.x] = to_fp32(dy_ptr[input_indexer.at(b, h, w, f)]);
							else
								gradient_tile[i * 32 + threadIdx.x] = 0.0f;
						}
						__syncthreads();

						for (int i = 0; i < TileSize; i++)
						{
							const int h = origin_h + threadIdx.y + i - Padding;
							if (is_inside(h, height))
							{
								Line<T, InputSize> inp;
								for (int j = 0; j < InputSize; j++)
								{
									const int w = origin_w + j - Padding;
									inp[j] = is_inside(w, width) ? to_fp32(x_ptr[input_indexer.at(blockIdx.y, h, w, f)]) : 0.0f;
								}
								const Line<T, TileSize> fil = convolution.load_filter(gradient_tile, i);
								convolution.accumulate(update_acc, inp, fil);
							}
						}
						__syncthreads();
					}

			const Indexer<4> update_indexer(gridDim.y, KernelSize, KernelSize, channels);
			for (int i = 0; i < KernelSize; i++)
				dw_ptr[update_indexer.at(blockIdx.y, threadIdx.y, i, f)] = update_acc[i];
		}
	}
	template<typename T>
	__global__ void kernel_sum_update(float beta, T *dst, float alpha, const float *src, int first_dim, int last_dim)
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
				local_sum += static_cast<float>(src[i * last_dim + last_dim_idx]);
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
		{
			float tmp = alpha * workspace[0][threadIdx.x];
			if (beta != 0.0f)
				tmp += beta * to_fp32(dst[last_dim_idx]);
			dst[last_dim_idx] = T(tmp);
		}
	}

	using namespace ml;

	template<typename T, int TileSize>
	void dispatch_dwconv_forward(mlContext_t context, float alpha, const mlTensor_t &x, const mlTensor_t &w, const mlTensor_t &b, float beta,
			mlTensor_t &y, bool invert)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(w.rank == 3);
		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int filter_size = w.dim[0];
		const int channels = w.dim[2];
		assert(w.dim[0] == w.dim[1]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		constexpr int channels_per_thread = std::is_same<T, vec2h>::value ? 2 : 1;
		const int channels_per_block = 32 * channels_per_thread;

		const int num_tiles_h = (height + TileSize - 1) / TileSize;

		dim3 blockDim(32, TileSize);
		dim3 gridDim(num_tiles_h, batch_size, (channels + channels_per_block - 1) / channels_per_block);
		switch (filter_size)
		{
			case 3:
				kernel_depthwise_conv_forward<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(beta, data<T>(y), alpha, data<T>(x), data<T>(w),
						data<T>(b), height, width, channels / channels_per_thread, invert);
				break;
			case 5:
				kernel_depthwise_conv_forward<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(beta, data<T>(y), alpha, data<T>(x), data<T>(w),
						data<T>(b), height, width, channels / channels_per_thread, invert);
				break;
			case 7:
				kernel_depthwise_conv_forward<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(beta, data<T>(y), alpha, data<T>(x), data<T>(w),
						data<T>(b), height, width, channels / channels_per_thread, invert);
				break;
			default:
				break;
		}
	}
	template<typename T, typename U, int TileSize>
	void dispatch_dwconv_update(mlContext_t context, float alpha, const mlTensor_t &x, const mlTensor_t &dy, float beta, mlTensor_t &dw)
	{
		assert(x.rank == 4);
		assert(dy.rank == 4);
		assert(dw.rank == 3);
		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int filter_size = dw.dim[0];
		const int channels = dw.dim[2];
		assert(dw.dim[0] == dw.dim[1]);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		float *workspace = getPointer<float>(ml::cuda_backend::Context::getWorkspace(context));
		const int workspace_size = ml::cuda_backend::Context::getWorkspaceSize(context);

		const int last_dim = filter_size * filter_size * channels;
		const int max_blocks = workspace_size / (sizeof(float) * last_dim);

		constexpr int channels_per_thread = std::is_same<T, vec2h>::value ? 2 : 1;
		const int channels_per_block = 32 * channels_per_thread;

		const int num_blocks = std::min(1024, std::min(batch_size, max_blocks));

		dim3 blockDim(32, filter_size);
		dim3 gridDim((channels + channels_per_block - 1) / channels_per_block, num_blocks);

		switch (filter_size)
		{
			case 3:
				kernel_depthwise_conv_update<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(workspace, data<T>(x), data<T>(dy), batch_size, height,
						width, channels / channels_per_thread);
				break;
			case 5:
				kernel_depthwise_conv_update<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(workspace, data<T>(x), data<T>(dy), batch_size, height,
						width, channels / channels_per_thread);
				break;
			case 7:
				kernel_depthwise_conv_update<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(workspace, data<T>(x), data<T>(dy), batch_size, height,
						width, channels / channels_per_thread);
				break;
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		blockDim = dim3(32, 32);
		gridDim = dim3((last_dim + channels_per_block - 1) / channels_per_block);
		kernel_sum_update<<<gridDim, blockDim, 0, stream>>>(beta, data<U>(dw), alpha, workspace, num_blocks, last_dim / channels_per_thread);
		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	void cuda_depthwise_conv_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y)
	{
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(w) % 2 == 0)
					dispatch_dwconv_forward<vec2h, 4>(context, alpha, x, w, b, beta, y, false);
				else
					dispatch_dwconv_forward<vec1h, 4>(context, alpha, x, w, b, beta, y, false);
				break;
			}
			case DTYPE_FLOAT32:
			{
				dispatch_dwconv_forward<vec1f, 4>(context, alpha, x, w, b, beta, y, false);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_depthwise_conv_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t w, float beta, mlTensor_t dx)
	{
		const mlTensor_t b = empty_tensor();
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(w) % 2 == 0)
					dispatch_dwconv_forward<vec2h, 4>(context, alpha, dy, w, b, beta, dx, true);
				else
					dispatch_dwconv_forward<vec1h, 4>(context, alpha, dy, w, b, beta, dx, true);
				break;
			}
			case DTYPE_FLOAT32:
			{
				dispatch_dwconv_forward<vec1f, 4>(context, alpha, dy, w, b, beta, dx, true);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_depthwise_conv_update(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t dy, float beta, mlTensor_t dw)
	{
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(is_fp32(dw) || is_fp16(dw));
				if (is_fp32(dw))
					dispatch_dwconv_update<vec1h, vec1f, 4>(context, alpha, x, dy, beta, dw);
				else
					dispatch_dwconv_update<vec1h, vec1h, 4>(context, alpha, x, dy, beta, dw);
				break;
			}
			case DTYPE_FLOAT32:
			{
				dispatch_dwconv_update<vec1f, vec1f, 4>(context, alpha, x, dy, beta, dw);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

