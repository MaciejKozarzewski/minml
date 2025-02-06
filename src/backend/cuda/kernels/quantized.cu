/*
 * quantized.cu
 *
 *  Created on: Feb 3, 2025
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
	template<typename T>
	__device__ T quantize(float x)
	{
	}
	template<>
	__device__ float quantize(float x)
	{
		return x;
	}
	template<>
	__device__ int8_t quantize(float x)
	{
		return static_cast<int8_t>(max(-128.0f, min(127.0f, x)));
	}

	struct Index2D
	{
			int8_t x = 0;
			int8_t y = 0;
	};

	__device__ bool is_inside(int h, int w, int height, int width)
	{
		return 0 <= h && h < height && 0 <= w && w < width;
	}

	template<int KernelSize, int OutputTile>
	struct Convolution1D
	{
			__device__ Line<int32_t, OutputTile> set(int32_t x) const
			{
				Line<int32_t, OutputTile> result;
				for (int i = 0; i < OutputTile; i++)
					result[i] = x;
				return result;
			}
			__device__ Line<int32_t, KernelSize + OutputTile - 1> load_input(const int32_t *src, int row) const
			{
				constexpr int N = KernelSize + OutputTile - 1;
				Line<int32_t, N> result;
				const int tid = (row * N + 0) * 32 + threadIdx.x;
				for (int i = 0; i < N; i++)
					result[i] = src[tid + i * 32];
				return result;
			}
			__device__ Line<int32_t, KernelSize> load_filter(const int32_t *src, int row) const
			{
				Line<int32_t, KernelSize> result;
				const int tid = (row * KernelSize + 0) * 32 + threadIdx.x;
				for (int i = 0; i < KernelSize; i++)
					result[i] = src[tid + i * 32];
				return result;
			}
			__device__ void store_output(int32_t *dst, int row, const Line<int32_t, OutputTile> &acc) const
			{
				const int tid = (row * OutputTile + 0) * 32 + threadIdx.x;
				for (int i = 0; i < OutputTile; i++)
					dst[tid + i * 32] = acc[i];
			}
			__device__ void accumulate(Line<int32_t, OutputTile> &acc, const Line<int32_t, KernelSize + OutputTile - 1> &input,
					const Line<int32_t, KernelSize> &filter) const
			{
				for (int i = 0; i < OutputTile; i++)
				{
					int32_t tmp = acc[i];
					for (int j = 0; j < KernelSize; j++)
						tmp += input[i + j] * filter[j];
					acc[i] = tmp;
				}
			}
	};

	template<>
	struct Convolution1D<7, 4>
	{
			__device__ Line<int32_t, 4> set(int32_t x) const
			{
				Line<int32_t, 4> result;
				result.x0 = x;
				result.x1 = x;
				result.x2 = x;
				result.x3 = x;
				return result;
			}
			__device__ Line<int32_t, 10> load_input(const int32_t *src, int row) const
			{
				const int tid = (row * 10 + 0) * 32 + threadIdx.x;
				Line<int32_t, 10> result;
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
			__device__ Line<int32_t, 7> load_filter(const int32_t *src, int row) const
			{
				const int tid = (row * 7 + 0) * 32 + threadIdx.x;
				Line<int32_t, 7> result;
				result.x0 = src[tid + 0 * 32];
				result.x1 = src[tid + 1 * 32];
				result.x2 = src[tid + 2 * 32];
				result.x3 = src[tid + 3 * 32];
				result.x4 = src[tid + 4 * 32];
				result.x5 = src[tid + 5 * 32];
				result.x6 = src[tid + 6 * 32];
				return result;
			}
			__device__ void store_output(int32_t *dst, int row, const Line<int32_t, 4> &acc) const
			{
				const int tid = (row * 4 + 0) * 32 + threadIdx.x;
				dst[tid + 0 * 32] = acc.x0;
				dst[tid + 1 * 32] = acc.x1;
				dst[tid + 2 * 32] = acc.x2;
				dst[tid + 3 * 32] = acc.x3;
			}
			__device__ void accumulate(Line<int32_t, 4> &acc, const Line<int32_t, 10> &input, const Line<int32_t, 7> &filter) const
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

	template<int TileSize, int KernelSize>
	__global__ void kernel_depthwise_conv_forward_int8(int8_t *output, const int8_t *input, const int8_t *weights, int height, int width,
			int channels, const float *scales, const float *bias)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == TileSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ int32_t input_tile[InputSize * InputSize * 32];
		__shared__ int32_t filter_tile[KernelSize * KernelSize * 32];
		__shared__ int32_t output_tile[TileSize * TileSize * 32];
		__shared__ float scale_tile[32];
		__shared__ float bias_tile[32];
		__shared__ Index2D indices[InputSize * InputSize];

		// load filters into shared memory
		const int f = blockIdx.z * blockDim.x + threadIdx.x;

		if (threadIdx.y == 0 and f < channels)
		{
			scale_tile[threadIdx.x] = (scales == nullptr) ? 1.0f : scales[f];
			bias_tile[threadIdx.x] = (bias == nullptr) ? 0.0f : bias[f];
		}

		for (int i = threadIdx.y; i < KernelSize * KernelSize; i += blockDim.y)
			if (f < channels)
				filter_tile[i * 32 + threadIdx.x] = weights[i * channels + f];
		// prepare indices table
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < InputSize * InputSize; i += blockDim.x * blockDim.y)
		{
			indices[i].x = i / InputSize;
			indices[i].y = i - indices[i].x * InputSize;
		}

		__syncthreads();

		const Convolution1D<KernelSize, TileSize> convolution;

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
					input_tile[i * 32 + threadIdx.x] = 0;
			}
			__syncthreads();

			Line<int32_t, TileSize> acc = convolution.set(0);
			for (int i = 0; i < KernelSize; i++)
			{
				const Line<int32_t, InputSize> inp = convolution.load_input(input_tile, i + threadIdx.y);
				const Line<int32_t, KernelSize> fil = convolution.load_filter(filter_tile, i);
				convolution.accumulate(acc, inp, fil);
			}
			convolution.store_output(output_tile, threadIdx.y, acc);

			for (int i = 0; i < TileSize; i++)
			{
				const int h = origin_h + threadIdx.y;
				const int w = origin_w + i;
				if (f < channels && is_inside(h, w, height, width))
				{
					const float tmp = static_cast<float>(output_tile[(threadIdx.y * TileSize + i) * 32 + threadIdx.x]);
					output[input_indexer.at(blockIdx.y, h, w, f)] = quantize<int8_t>(tmp * scale_tile[threadIdx.x] + bias_tile[threadIdx.x]);
				}
			}
		}
	}

	template<typename T>
	__global__ void kernel_scale_shift_act(T *output, const int32_t *input, const float *scales, const float *bias, int first_dim, int last_dim,
			ml::mlActivationType_t act, const int8_t *ext, float ext_scale, float ext_shift)
	{
		const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
		const float scale = scales[tid_x];
		const float shift = bias[tid_x];

		for (int i = tid_y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			float x = static_cast<float>(input[i * last_dim + tid_x]) * scale + shift;
			if (ext != nullptr)
				x += static_cast<float>(ext[i * last_dim + tid_x]) * ext_scale + ext_shift;

			if (act == ml::ACTIVATION_RELU)
				x = max(0.0f, x);

			output[i * last_dim + tid_x] = quantize<T>(x);
		}
	}
	template<typename T>
	__global__ void kernel_dequantize(T *output, const int8_t *input, float scale, float shift, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			output[i] = static_cast<T>(static_cast<float>(input[i]) * scale + shift);
	}

	int get_block_size(int size) noexcept
	{
		for (int i = 16; i >= 1; i /= 2)
			if (size % i == 0)
				return i;
		return 1;
	}

	template<typename T>
	__launch_bounds__(256, 8)
	__global__ void kernel_receptive_fields(const void *input, void *output, int4 input_shape, int kernel_size, T padding_value)
	{
		int input_height = input_shape.y + kernel_size - 1;
		int input_width = input_shape.z + kernel_size - 1;
		int filters = input_shape.w;

		int volume = input_shape.x * input_height * input_width * filters;
		for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < volume; tid += gridDim.x * blockDim.x)
		{
			int in_f = tid;
			int in_b = in_f / (input_height * input_width * filters);
			in_f -= in_b * input_height * input_width * filters;
			int in_h = in_f / (input_width * filters);
			in_f -= in_h * input_width * filters;
			int in_w = in_f / filters;
			in_f -= in_w * filters;

			in_h = in_h - kernel_size / 2;
			in_w = in_w - kernel_size / 2;

			T loaded = padding_value;
			const int idx = ((in_b * input_shape.y + in_h) * input_shape.z + in_w) * filters + in_f;
			if (in_h >= 0 && in_h < input_shape.y && in_w >= 0 && in_w < input_shape.z)
				loaded = reinterpret_cast<const T*>(input)[idx];

			for (int i = 0; i < kernel_size; i++)
				for (int j = 0; j < kernel_size; j++)
				{
					const int x = in_h + i - kernel_size / 2;
					const int y = in_w + j - kernel_size / 2;
					int offset = i * kernel_size + j;
					offset = (kernel_size * kernel_size - 1) - offset;
					if (x >= 0 && x < input_shape.y && y >= 0 && y < input_shape.z)
						reinterpret_cast<T*>(output)[(((in_b * input_shape.y + x) * input_shape.z + y) * kernel_size * kernel_size + offset) * filters
								+ in_f] = loaded;
				}
		}
	}
}

namespace ml
{
	void cuda_dequantize(mlContext_t context, mlDataType_t dtype, const void *input, void *output, int elements, float scale, float shift)
	{
		cudaStream_t stream = cuda::Context::getStream(context);
		dim3 blockDim(256);
		dim3 gridDim(std::min(1024, (elements + 255) / 256));

		switch (dtype)
		{
			case DTYPE_FLOAT64:
				kernel_dequantize<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<int8_t>(input), elements, scale, shift);
				break;
			case DTYPE_FLOAT32:
				kernel_dequantize<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<int8_t>(input), elements, scale, shift);
				break;
			case DTYPE_FLOAT16:
				kernel_dequantize<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<int8_t>(input), elements, scale, shift);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_quantized_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, const void *scales, const void *bias, void *output)
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

		if (dtype == DTYPE_INT32)
		{
			switch (filter_size)
			{
				case 3:
					kernel_depthwise_conv_forward_int8<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels, getPointer<float>(scales),
							getPointer<float>(bias));
					break;
				case 5:
					kernel_depthwise_conv_forward_int8<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels, getPointer<float>(scales),
							getPointer<float>(bias));
					break;
				case 7:
					kernel_depthwise_conv_forward_int8<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels, getPointer<float>(scales),
							getPointer<float>(bias));
					break;
				default:
					break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_quantized_scale_shift_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *scales,
			const void *bias, mlActivationType_t act, const void *ext, float ext_scale, float ext_shift)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(32, 8);
		dim3 gridDim((last_dim + 31) / 32, std::min(std::max(1, first_dim / 8), 1024));

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), ext_scale, ext_shift);
				break;
			case DTYPE_FLOAT32:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), ext_scale, ext_shift);
				break;
			case DTYPE_FLOAT64:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), ext_scale, ext_shift);
				break;
			case DTYPE_INT8:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), ext_scale, ext_shift);
				break;

		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_create_receptive_fields(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, void *output, const void *input,
			int kernel_size)
	{
		const int last_dim = get_last_dim(input_shape) * size_of(dtype);

		cudaStream_t stream = cuda::Context::getStream(context);
		dim3 blockSize(256);
		dim3 gridSize(std::min(2048, (volume(input_shape) + 255) / 256));

		const int block_size = get_block_size(last_dim);

		const int4 shape { input_shape.dim[0], input_shape.dim[1], input_shape.dim[2], last_dim / block_size };

		switch (block_size)
		{
			default:
			case 1:
				kernel_receptive_fields<int8_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, 0);
				break;
			case 2:
				kernel_receptive_fields<int16_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, 0);
				break;
			case 4:
				kernel_receptive_fields<int32_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, 0);
				break;
			case 8:
				kernel_receptive_fields<int2> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, int2 { 0, 0 });
				break;
			case 16:
				kernel_receptive_fields<int4> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, int4 { 0, 0, 0, 0 });
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace */

