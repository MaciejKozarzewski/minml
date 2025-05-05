/*
 * conversion.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"

#include "../helpers/indexers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cassert>

namespace
{
	using namespace ml;

	template<typename T>
	__device__ T one()
	{
		return static_cast<T>(1.0f);
	}
	template<typename T>
	__device__ T zero()
	{
		return static_cast<T>(0.0f);
	}

	template<typename T>
	__global__ void kernel_unpack_input(T *output, const uint32_t *input, int first_dim, int last_dim)
	{
		const int stride = (last_dim + 31) / 32;
		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const int int_idx = j / 32;
				const int bit_idx = j % 32;
				const uint32_t value = input[i * stride + int_idx] >> bit_idx;
				output[i * last_dim + j] = (value & 1) ? one<T>() : zero<T>();
			}
	}

	template<typename T, typename U>
	__global__ void kernel_convert(T *output, const U *input, int length)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			output[i] = static_cast<T>(static_cast<float>(input[i]));
	}

	template<typename T, int Rows, int Columns>
	__global__ void kernel_transpose_021(T *output, const T *input, int dim0, int dim1, int dim2)
	{
		__shared__ T workspace[Rows * Columns];
		constexpr int Padding = 4 / sizeof(T);

		Indexer<3> src_indexer(dim0, dim1, dim2);
		for (int row = threadIdx.y; row < Rows; row += blockDim.y)
			for (int col = threadIdx.x; col < Columns; col += blockDim.x)
			{
				const int d0 = blockIdx.x;
				const int d1 = blockIdx.y * Rows + row;
				const int d2 = blockIdx.z * Columns + col;
				if (d1 < dim1 && d2 < dim2)
				{
					const int idx = row * Columns + (row * Padding + col) % Columns;
					workspace[idx] = input[src_indexer.at(d0, d1, d2)];
				}
			}
		__syncthreads();

		Indexer<3> dst_indexer(dim0, dim2, dim1);
		for (int col = threadIdx.y; col < Columns; col += blockDim.y)
			for (int row = threadIdx.x; row < Rows; row += blockDim.x)
			{
				const int d0 = blockIdx.x;
				const int d1 = blockIdx.y * Rows + row;
				const int d2 = blockIdx.z * Columns + col;
				if (d1 < dim1 && d2 < dim2)
				{
					const int idx = row * Columns + (row * Padding + col) % Columns;
					output[dst_indexer.at(d0, d2, d1)] = workspace[idx];
				}
			}
	}

	__global__ void kernel_calculate_space2depth_offsets(int *output, int input_height, int input_width, int input_channels, int patch_size_h,
			int patch_size_w)
	{
		const int num_patches_h = (input_height + patch_size_h - 1) / patch_size_h;
		const int num_patches_w = (input_width + patch_size_w - 1) / patch_size_w;
		const int num_channels = input_channels;

		const int elements = num_channels * patch_size_h * patch_size_w * num_patches_w * num_patches_h;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			const int stride0 = 1;
			const int stride1 = 1 * num_channels;
			const int stride2 = 1 * num_channels * patch_size_w;
			const int stride3 = 1 * num_channels * patch_size_w * patch_size_h;
			const int stride4 = 1 * num_channels * patch_size_w * patch_size_h * num_patches_w;

			const int channel = (i / stride0) % num_channels;
			const int in_patch_w = (i / stride1) % patch_size_w;
			const int in_patch_h = (i / stride2) % patch_size_h;
			const int patch_idx_w = (i / stride3) % num_patches_w;
			const int patch_idx_h = (i / stride4) % num_patches_h;

			if ((patch_idx_h * patch_size_h + in_patch_h) < input_height && (patch_idx_w * patch_size_w + in_patch_w) < input_width)
			{
				const Indexer<3> input_indexer(input_height, input_width, num_channels);
				output[i] = input_indexer.at(patch_idx_h * patch_size_h + in_patch_h, patch_idx_w * patch_size_w + in_patch_w, channel);
			}
			else
				output[i] = -1;
		}
	}
	template<typename T>
	__global__ void kernel_space_to_depth(T *output, const T *input, const int *offsets, int batch_size, int output_elements, int input_elements)
	{
		for (int i = blockIdx.y; i < batch_size; i += gridDim.y)
			for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < output_elements; j += gridDim.x * blockDim.x)
			{
				const int idx = offsets[j];
				if (idx == -1)
					output[i * output_elements + j] = static_cast<T>(0);
				else
					output[i * output_elements + j] = input[i * input_elements + idx];
			}
	}
	template<typename T>
	__global__ void kernel_depth_to_space(T *output, const T *input, const int *offsets, int batch_size, int output_elements, int input_elements)
	{
		for (int i = blockIdx.y; i < batch_size; i += gridDim.y)
			for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < input_elements; j += gridDim.x * blockDim.x)
			{
				const int idx = offsets[j];
				if (idx != -1)
					output[i * output_elements + idx] = input[i * input_elements + j];
			}
	}
	int get_patch_size(int smaller, int larger) noexcept
	{
		assert(smaller <= larger);
		for (int i = 1;; i++)
		{
			const int tmp = (larger + i - 1) / i;
			if (tmp == smaller)
				return i;
			if (tmp < smaller)
				break;
		}
		return 0;
	}

	template<typename T>
	void convert_helper(cudaStream_t stream, void *dst, const void *src, mlDataType_t src_dtype, int elements)
	{
		const dim3 blockDim(256);
		const dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, 256);
		switch (src_dtype)
		{
			case DTYPE_FLOAT16:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<half>(src), elements);
				break;
			case DTYPE_FLOAT32:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<float>(src), elements);
				break;
			case DTYPE_FLOAT64:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<double>(src), elements);
				break;
			case DTYPE_UINT8:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<uint8_t>(src), elements);
				break;
			case DTYPE_INT8:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<int8_t>(src), elements);
				break;
			case DTYPE_INT16:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<int16_t>(src), elements);
				break;
			case DTYPE_INT32:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(getPointer<T>(dst), getPointer<int32_t>(src), elements);
				break;
		}
	}

}

namespace ml
{
	void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		dim3 blockDim = std::min(last_dim, 256);
		dim3 gridDim = std::min(first_dim, 1024);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT64:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_UINT8:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<uint8_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT8:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT16:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<int16_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT32:
				kernel_unpack_input<<<gridDim, blockDim, 0, stream>>>(getPointer<int32_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		if (elements == 0)
			return;
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		if (dst_dtype == src_dtype && dst != src)
		{ // same type, different locations, can just copy memory
			cudaError_t status = cudaMemcpyAsync(dst, src, elements * size_of(dst_dtype), cudaMemcpyDeviceToDevice, stream);
			assert(status == cudaSuccess);
			return;
		}

		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				convert_helper<half>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_FLOAT32:
				convert_helper<float>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_FLOAT64:
				convert_helper<double>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_UINT8:
				convert_helper<uint8_t>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_INT8:
				convert_helper<int8_t>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_INT16:
				convert_helper<int16_t>(stream, dst, src, src_dtype, elements);
				break;
			case DTYPE_INT32:
				convert_helper<int32_t>(stream, dst, src, src_dtype, elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		assert(input != output);

		dim3 blockDim(64, 8);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_INT8:
			case DTYPE_UINT8:
			{
				dim3 gridDim(shape.dim[0], (shape.dim[1] + 128 - 1) / 128, (shape.dim[2] + 128 - 1) / 128);
				kernel_transpose_021<uint8_t, 128, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint8_t>(output), getPointer<uint8_t>(input),
						shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			}
			case DTYPE_INT16:
			case DTYPE_FLOAT16:
			{
				dim3 gridDim(shape.dim[0], (shape.dim[1] + 128 - 1) / 128, (shape.dim[2] + 64 - 1) / 64);
				kernel_transpose_021<uint16_t, 128, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint16_t>(output), getPointer<uint16_t>(input),
						shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			}
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
			{
				dim3 gridDim(shape.dim[0], (shape.dim[1] + 64 - 1) / 64, (shape.dim[2] + 64 - 1) / 64);
				kernel_transpose_021<uint32_t, 64, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint32_t>(output), getPointer<uint32_t>(input),
						shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_space_to_depth(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		const int batch_size = get_first_dim(input_shape);
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int patch_size_h = get_patch_size(output_shape.dim[1], input_shape.dim[1]);
		const int patch_size_w = get_patch_size(output_shape.dim[2], input_shape.dim[2]);
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = get_last_dim(input_shape);
		const int channels_out = get_last_dim(output_shape);
		assert(channels_in * patch_size_h * patch_size_w == channels_out);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		const int output_elements = output_shape.dim[1] * output_shape.dim[2] * output_shape.dim[3];
		dim3 blockDim(256);
		dim3 gridDim((output_elements + blockDim.x - 1) / blockDim.x);

		int *offsets = getPointer<int>(ml::cuda_backend::Context::getWorkspace(context));
		assert(output_elements * sizeof(int) <= ml::cuda_backend::Context::getWorkspaceSize(context));

		kernel_calculate_space2depth_offsets<<<gridDim, blockDim, 0, stream>>>(offsets, height, width, channels_in, patch_size_h, patch_size_w);
		assert(cudaGetLastError() == cudaSuccess);

		const int input_elements = input_shape.dim[1] * input_shape.dim[2] * input_shape.dim[3];

		dim3 blockDim2(256);
		dim3 gridDim2((output_elements + blockDim2.x - 1) / blockDim2.x, batch_size);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_space_to_depth<<<gridDim2, blockDim2, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), offsets, batch_size,
						output_elements, input_elements);
				break;
			case DTYPE_FLOAT32:
				kernel_space_to_depth<<<gridDim2, blockDim2, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), offsets, batch_size,
						output_elements, input_elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_depth_to_space(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		const int batch_size = get_first_dim(input_shape);
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int patch_size_h = get_patch_size(input_shape.dim[1], output_shape.dim[1]);
		const int patch_size_w = get_patch_size(input_shape.dim[2], output_shape.dim[2]);
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = get_last_dim(input_shape);
		const int channels_out = get_last_dim(output_shape);
		assert(channels_out * patch_size_h * patch_size_w == channels_in);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		const int input_elements = input_shape.dim[1] * input_shape.dim[2] * input_shape.dim[3];
		dim3 blockDim(256);
		dim3 gridDim((input_elements + blockDim.x - 1) / blockDim.x);

		int *offsets = getPointer<int>(ml::cuda_backend::Context::getWorkspace(context));
		assert(input_elements * sizeof(int) <= ml::cuda_backend::Context::getWorkspaceSize(context));

		kernel_calculate_space2depth_offsets<<<gridDim, blockDim, 0, stream>>>(offsets, height, width, channels_out, patch_size_h, patch_size_w);
		assert(cudaGetLastError() == cudaSuccess);

		const int output_elements = output_shape.dim[1] * output_shape.dim[2] * output_shape.dim[3];

		dim3 blockDim2(256);
		dim3 gridDim2((input_elements + blockDim2.x - 1) / blockDim2.x, batch_size);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_depth_to_space<<<gridDim2, blockDim2, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), offsets, batch_size,
						output_elements, input_elements);
				break;
			case DTYPE_FLOAT32:
				kernel_depth_to_space<<<gridDim2, blockDim2, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), offsets, batch_size,
						output_elements, input_elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

