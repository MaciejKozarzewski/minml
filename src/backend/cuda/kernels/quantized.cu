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
#include <cstring>
#include <type_traits>

namespace
{
	using namespace vectors2;

	struct Quantizer
	{
			float scale = 1.0f;
			float shift = 0.0f;

			__host__ Quantizer(const ml::mlQuantizationData_t &qd) :
					scale(qd.scale),
					shift(qd.shift)
			{
			}
			__device__ float to_fp32(int8_t x) const
			{
				return static_cast<float>(x) * scale + shift;
			}
			__device__ int8_t to_int8(float x) const
			{
				return static_cast<int8_t>(max(-128.0f, min(127.0f, round((x - shift) / scale))));
			}
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
	__global__ void kernel_depthwise_conv_forward_int8(int8_t *output, Quantizer output_quantizer, const int8_t *input, const int8_t *weights,
			int height, int width, int channels, const float *scales, const float *bias, int32_t padding_value)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == TileSize);
		constexpr int InputSize = TileSize + KernelSize - 1;
		constexpr int Padding = (KernelSize - 1) / 2;
		__shared__ int32_t filter_tile[KernelSize * KernelSize * 32];
		__shared__ float scale_tile[32];
		__shared__ float bias_tile[32];

		const int f = blockIdx.z * blockDim.x + threadIdx.x;

		if (threadIdx.y == 0 and f < channels)
		{
			scale_tile[threadIdx.x] = (scales == nullptr) ? 1.0f : scales[f];
			bias_tile[threadIdx.x] = (bias == nullptr) ? 0.0f : bias[f];
		}

		for (int i = threadIdx.y; i < KernelSize * KernelSize; i += blockDim.y)
			if (f < channels)
				filter_tile[i * 32 + threadIdx.x] = weights[i * channels + f];
		__syncthreads();

		const Convolution1D<KernelSize, TileSize> convolution;
		const Indexer<4> input_indexer(gridDim.y, height, width, channels);

		for (int origin_w = 0; origin_w < width; origin_w += TileSize)
		{
			const int origin_h = TileSize * blockIdx.x;
			Line<int32_t, TileSize> acc = convolution.set(bias_tile[threadIdx.x]);
			for (int k = 0; k < KernelSize; k++)
			{
				Line<int32_t, InputSize> inp;
				for (int i = 0; i < InputSize; i++)
				{
					const int h = origin_h + threadIdx.y + k - Padding;
					const int w = origin_w + i - Padding;
					if (f < channels && is_inside(h, w, height, width))
						inp[i] = input[input_indexer.at(blockIdx.y, h, w, f)];
					else
						inp[i] = padding_value;
				}

				const Line<int32_t, KernelSize> fil = convolution.load_filter(filter_tile, k);
				convolution.accumulate(acc, inp, fil);
			}
			for (int i = 0; i < TileSize; i++)
			{
				const int h = origin_h + threadIdx.y;
				const int w = origin_w + i;
				if (f < channels && is_inside(h, w, height, width))
				{
					const float tmp = acc[i] * scale_tile[threadIdx.x] + bias_tile[threadIdx.x];
					output[input_indexer.at(blockIdx.y, h, w, f)] = output_quantizer.to_int8(tmp);
				}
			}
		}
	}

	template<typename T>
	__global__ void kernel_scale_shift_act(T *output, Quantizer output_quantizer, const int32_t *input, const float *scales, const float *bias,
			int first_dim, int last_dim, ml::mlActivationType_t act, const int8_t *ext, Quantizer ext_quantizer)
	{
		const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
		if (tid_x < last_dim)
		{
			const float scale = scales[tid_x];
			const float shift = bias[tid_x];

			for (int i = tid_y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				float x = input[i * last_dim + tid_x] * scale + shift;
				if (ext != nullptr)
					x += ext_quantizer.to_fp32(ext[i * last_dim + tid_x]);

				switch (act)
				{
					case ml::ACTIVATION_LINEAR:
						break;
					case ml::ACTIVATION_SIGMOID:
						x = 1.0f / (1.0f + expf(-x));
						break;
					case ml::ACTIVATION_TANH:
						x = tanhf(x);
						break;
					case ml::ACTIVATION_RELU:
						x = max(0.0f, x);
						break;
				}

				if (std::is_same<T, float>::value)
					output[i * last_dim + tid_x] = x;
				if (std::is_same<T, int8_t>::value)
					output[i * last_dim + tid_x] = output_quantizer.to_int8(x);
			}
		}
	}
	template<typename T>
	__global__ void kernel_dequantize(T *output, const int8_t *input, float scale, float shift, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			output[i] = static_cast<T>(static_cast<float>(input[i]) * scale + shift);
	}

	template<typename T>
	__global__ void kernel_receptive_fields(const void *input, void *output, int4 input_shape, int kernel_size, bool invert, T padding_value)
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

			in_h = in_h - (kernel_size - 1) / 2;
			in_w = in_w - (kernel_size - 1) / 2;

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
					if (invert == false)
						offset = (kernel_size * kernel_size - 1) - offset;
					if (x >= 0 && x < input_shape.y && y >= 0 && y < input_shape.z)
						reinterpret_cast<T*>(output)[(((in_b * input_shape.y + x) * input_shape.z + y) * kernel_size * kernel_size + offset) * filters
								+ in_f] = loaded;
				}
		}
	}

	__device__ float to_fp8_and_back(float x)
	{
		const uint32_t tmp = reinterpret_cast<uint32_t*>(&x)[0] & 0xFFF00000u;
		return reinterpret_cast<const float*>(&tmp)[0];
	}

	__device__ vec<float, 1> emulate_fp8(vec<float, 1> x)
	{
		x.x0 = to_fp8_and_back(x.x0);
		return x;
	}
	__device__ vec<float, 4> emulate_fp8(vec<float, 4> x)
	{
		x.x0 = to_fp8_and_back(x.x0);
		x.x1 = to_fp8_and_back(x.x1);
		x.x2 = to_fp8_and_back(x.x2);
		x.x3 = to_fp8_and_back(x.x3);
		return x;
	}
	__device__ vec<float, 1> emulate_fp16(vec<float, 1> x)
	{
		x.x0 = static_cast<float>(static_cast<half>(x.x0));
		return x;
	}
	__device__ vec<float, 4> emulate_fp16(vec<float, 4> x)
	{
		x.x0 = static_cast<float>(static_cast<half>(x.x0));
		x.x1 = static_cast<float>(static_cast<half>(x.x1));
		x.x2 = static_cast<float>(static_cast<half>(x.x2));
		x.x3 = static_cast<float>(static_cast<half>(x.x3));
		return x;
	}
	__device__ vec<float, 1> emulate_int8(vec<float, 1> x, Quantizer q)
	{
		x.x0 = q.to_fp32(q.to_int8(x.x0));
		return x;
	}
	__device__ vec<float, 4> emulate_int8(vec<float, 4> x, Quantizer q)
	{
		x.x0 = q.to_fp32(q.to_int8(x.x0));
		x.x1 = q.to_fp32(q.to_int8(x.x1));
		x.x2 = q.to_fp32(q.to_int8(x.x2));
		x.x3 = q.to_fp32(q.to_int8(x.x3));
		return x;
	}

	template<int N>
	__global__ void kernel_emulate_low_precision_fp8(float *dst, const float *src, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = tid * N; i < elements; i += stride * N)
		{
			const vec<float, N> x(src + i);
			const vec<float, N> y = emulate_fp8(x);
			y.store(dst + i);
		}
	}
	template<int N>
	__global__ void kernel_emulate_low_precision_fp16(float *dst, const float *src, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = tid * N; i < elements; i += stride * N)
		{
			const vec<float, N> x(src + i);
			const vec<float, N> y = emulate_fp16(x);
			y.store(dst + i);
		}
	}
	template<int N>
	__global__ void kernel_emulate_low_precision_int8(float *dst, const float *src, int elements, Quantizer q)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		for (int i = tid * N; i < elements; i += stride * N)
		{
			const vec<float, N> x(src + i);
			const vec<float, N> y = emulate_int8(x, q);
			y.store(dst + i);
		}
	}

	template<typename T>
	T get_padding(const void *ptr)
	{
		if (ptr == nullptr)
			return T { };
		else
		{
			T result;
			std::memcpy(&result, ptr, sizeof(T));
			return result;
		}
	}
}

namespace ml
{
	void cuda_emulate_low_precision(mlContext_t context, mlShape_t shape, mlDataType_t dtype, void *dst, const void *src, mlQuantizationData_t qd)
	{
		const int elements = volume(shape);
		const int vect = (elements % 4 == 0) ? 4 : 1;
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(elements, blockDim.x * vect);

		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_FLOAT8:
				if (vect == 4)
					kernel_emulate_low_precision_fp8<4> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements);
				else
					kernel_emulate_low_precision_fp8<1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements);
				break;
			case DTYPE_FLOAT16:
				if (vect == 4)
					kernel_emulate_low_precision_fp16<4> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements);
				else
					kernel_emulate_low_precision_fp16<1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements);
				break;
			case DTYPE_INT8:
				if (vect == 4)
					kernel_emulate_low_precision_int8<4> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements,
							Quantizer(qd));
				else
					kernel_emulate_low_precision_int8<1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src), elements,
							Quantizer(qd));
				break;
			default:
				const cudaError_t err = cudaMemcpyAsync(dst, src, elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
				assert(err == cudaSuccess);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
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
			const void *input, const void *weights, const void *scales, const void *bias, void *output, mlQuantizationData_t output_qd,
			int padding_value)
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

		if (dtype == DTYPE_INT8)
		{
			switch (filter_size)
			{
				case 3:
					kernel_depthwise_conv_forward_int8<TileSize, 3> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							Quantizer(output_qd), getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels,
							getPointer<float>(scales), getPointer<float>(bias), padding_value);
					break;
				case 5:
					kernel_depthwise_conv_forward_int8<TileSize, 5> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							Quantizer(output_qd), getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels,
							getPointer<float>(scales), getPointer<float>(bias), padding_value);
					break;
				case 7:
					kernel_depthwise_conv_forward_int8<TileSize, 7> <<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output),
							Quantizer(output_qd), getPointer<int8_t>(input), getPointer<int8_t>(weights), height, width, channels,
							getPointer<float>(scales), getPointer<float>(bias), padding_value);
					break;
				default:
					break;
			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_quantized_scale_shift_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, mlQuantizationData_t output_qd,
			const void *input, const void *scales, const void *bias, mlActivationType_t act, const void *ext, mlQuantizationData_t ext_qd)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		dim3 blockDim(32, 8);
		dim3 gridDim((last_dim + 31) / 32, std::min(std::max(1, first_dim / 8), 1024));

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), Quantizer(output_qd), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), Quantizer(ext_qd));
				break;
			case DTYPE_FLOAT32:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), Quantizer(output_qd), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), Quantizer(ext_qd));
				break;
			case DTYPE_FLOAT64:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), Quantizer(output_qd), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), Quantizer(ext_qd));
				break;
			case DTYPE_INT8:
				kernel_scale_shift_act<<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(output), Quantizer(output_qd), getPointer<int32_t>(input),
						getPointer<float>(scales), getPointer<float>(bias), first_dim, last_dim, act, getPointer<int8_t>(ext), Quantizer(ext_qd));
				break;

		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_im2row(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, void *output, const void *input, int kernel_size, bool invert,
			const void *padding)
	{
		const int last_dim = get_last_dim(input_shape);

		cudaStream_t stream = cuda::Context::getStream(context);
		dim3 blockSize(256);
		dim3 gridSize(std::min(2048, (volume(input_shape) + 255) / 256));

		const int4 shape { input_shape.dim[0], input_shape.dim[1], input_shape.dim[2], last_dim };

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_receptive_fields<half> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<half>(padding));
				break;
			case DTYPE_FLOAT32:
				kernel_receptive_fields<float> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<float>(padding));
				break;
			case DTYPE_FLOAT64:
				kernel_receptive_fields<double> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<double>(padding));
				break;
			case DTYPE_UINT8:
				kernel_receptive_fields<uint8_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<uint8_t>(padding));
				break;
			case DTYPE_INT8:
				kernel_receptive_fields<int8_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<int8_t>(padding));
				break;
			case DTYPE_INT16:
				kernel_receptive_fields<int16_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<int16_t>(padding));
				break;
			case DTYPE_INT32:
				kernel_receptive_fields<int32_t> <<<gridSize, blockSize, 0, stream>>>(input, output, shape, kernel_size, invert,
						get_padding<int32_t>(padding));
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace */

