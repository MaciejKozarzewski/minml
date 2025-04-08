/*
 * winograd.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "winograd_transforms.cuh"
#include "../utils.hpp"

#include "../helpers/indexers.cuh"
#include "../helpers/lines_and_tiles.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>

namespace
{
	using namespace ml;

	__device__ bool is_inside(int h, int w, int height, int width)
	{
		return 0 <= h && h < height && 0 <= w && w < width;
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_weights(T *__restrict__ matrices, const T *__restrict__ weights, int output_filters, int input_filters,
			bool invert)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<T, KernelSize, KernelSize> tile;
		for (int f = threadIdx.x; f < input_filters; f += blockDim.x)
		{
			const Indexer<4> weights_indexer(output_filters, KernelSize, KernelSize, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					T tmp;
					if (invert)
						tmp = weights[weights_indexer.at(blockIdx.y, KernelSize - 1 - row, KernelSize - 1 - col, f)];
					else
						tmp = weights[weights_indexer.at(blockIdx.y, row, col, f)];
					tile.at(col, row) = tmp;
				}

			const Indexer<4> matrices_indexer(TileSize, TileSize, output_filters, input_filters);

			const Transform<TransformType::WEIGHT, KernelSize, TransformSize, T> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<T, KernelSize> line;
				for (int col = 0; col < KernelSize; col++)
					line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

				for (int col = 0; col < TileSize; col++)
					matrices[matrices_indexer.at(row, col, blockIdx.y, f)] = transform(col, line);
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_input(T *__restrict__ matrices, const T *__restrict__ input, int batch_size, int height, int width,
			int input_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;
		constexpr int Padding = KernelSize / 2;

		Tile<T, TileSize, TileSize> tile;
		for (int f = threadIdx.x; f < input_filters; f += blockDim.x)
		{
			const Indexer<4> input_indexer(batch_size, height, width, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					const int h = TransformSize * blockIdx.x - Padding + row;
					const int w = TransformSize * blockIdx.y - Padding + col;
					tile.at(col, row) = is_inside(h, w, height, width) ? input[input_indexer.at(blockIdx.z, h, w, f)] : get<T>(0.0f);
				}

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;

			const Indexer<4> matrices_indexer(TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, input_filters);
			const Transform<TransformType::INPUT, KernelSize, TransformSize, T> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<T, TileSize> line;
				for (int col = 0; col < TileSize; col++)
					line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

				for (int col = 0; col < TileSize; col++)
					matrices[matrices_indexer.at(row, col, tile_index, f)] = transform(col, line);
			}
		}
	}
	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_output(const T *__restrict__ matrices, T *__restrict__ output, const T *__restrict__ add,
			const T *__restrict__ bias, mlActivationType_t activation, int batch_size, int height, int width, int output_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<T, TileSize, TileSize> tile;
		for (int f = threadIdx.x; f < output_filters; f += blockDim.x)
		{
			const T bias_value = (bias != nullptr) ? bias[f] : get<T>(0.0f);

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			const Indexer<4> matrices_indexer(TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, output_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
					tile.at(col, row) = matrices[matrices_indexer.at(row, col, tile_index, f)];

			const Indexer<4> output_indexer(batch_size, height, width, output_filters);
			const Transform<TransformType::OUTPUT, KernelSize, TransformSize, T> transform;
			for (int row = 0; row < TransformSize; row++)
			{
				const int h = TransformSize * blockIdx.x + row;
				if (h < height)
				{
					Line<T, TileSize> line;
					for (int col = 0; col < TileSize; col++)
						line[col] = transform(row, tile.get_row(col));

					for (int col = 0; col < TransformSize; col++)
					{
						const int w = TransformSize * blockIdx.y + col;
						if (w < width)
						{
							T tmp = transform(col, line);

							if (add != nullptr)
								tmp += add[output_indexer.at(blockIdx.z, h, w, f)];
							if (bias != nullptr)
								tmp += bias_value;
							switch (activation)
							{
								case ACTIVATION_SIGMOID:
									tmp = ml::internal::sigmoid(tmp);
									break;
								case ACTIVATION_TANH:
									tmp = ml::internal::tanh(tmp);
									break;
								case ACTIVATION_RELU:
									tmp = ml::internal::relu(tmp);
									break;
								case ACTIVATION_LEAKY_RELU:
									tmp = ml::internal::leaky_relu(tmp);
									break;
							}

							output[output_indexer.at(blockIdx.z, h, w, f)] = tmp;
						}
					}
				}
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_gradient(T *__restrict__ matrices, const T *__restrict__ gradient, int batch_size, int height, int width,
			int output_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<T, TransformSize, TransformSize> tile;
		for (int f = threadIdx.x; f < output_filters; f += blockDim.x)
		{
			const Indexer<4> gradient_indexer(batch_size, height, width, output_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					const int h = TransformSize * blockIdx.x + row;
					const int w = TransformSize * blockIdx.y + col;
					tile.at(col, row) = is_inside(h, w, height, width) ? gradient[gradient_indexer.at(blockIdx.z, h, w, f)] : get<T>(0.0f);
				}

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			const Indexer<4> matrices_indexer(TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, output_filters);
			const Transform<TransformType::GRADIENT, KernelSize, TransformSize, T> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<T, TransformSize> line;
				for (int col = 0; col < TransformSize; col++)
					line[col] = transform(row, tile.get_row(col));

				for (int col = 0; col < TileSize; col++)
					matrices[matrices_indexer.at(row, col, tile_index, f)] = transform(col, line);
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T, typename U>
	__global__ void kernel_transform_update(const T *__restrict__ matrices, U *__restrict__ update, int output_filters, int input_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<T, TileSize, TileSize> tile;
		for (int f = threadIdx.x; f < input_filters; f += blockDim.x)
		{
			const Indexer<4> matrices_indexer(TileSize, TileSize, output_filters, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
					tile.at(col, row) = matrices[matrices_indexer.at(row, col, blockIdx.y, f)];

			const Indexer<4> update_indexer(output_filters, KernelSize, KernelSize, input_filters);
			const Transform<TransformType::UPDATE, KernelSize, TransformSize, T> transform;
			for (int row = 0; row < KernelSize; row++)
			{
				Line<T, TileSize> line;
				for (int col = 0; col < TileSize; col++)
					line[col] = transform(row, tile.get_row(col));

				for (int col = 0; col < KernelSize; col++)
					update[update_indexer.at(blockIdx.y, row, col, f)] = transform(col, line);
			}
		}
	}

	/*
	 * host code
	 */
	template<typename T>
	int vector_length()
	{
		return 1;
	}
	template<>
	int vector_length<half2>()
	{
		return 2;
	}
	int get_number_of_tiles(int dim, int transform_size)
	{
		return (dim + transform_size - 1) / transform_size;
	}
	int get_kernel_size(const mlTensor_t &matrices, int tile_size)
	{
		assert(matrices.rank == 3);
		return std::sqrt(matrices.dim[0]) + 1 - tile_size;
	}

	template<typename T>
	void launch_weight_transform(mlContext_t context, int tile_size, const mlTensor_t w, mlTensor_t matrices, bool invert)
	{
		const int filters_out = w.dim[0];
		const int filters_in = w.dim[3] / vector_length<T>();

		const int kernel_size = get_kernel_size(matrices, tile_size);

		dim3 blockSize(std::min(128, filters_in));
		dim3 gridSize(1, filters_out);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_weights<3, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(w), filters_out, filters_in, invert);
			if (tile_size == 4)
				kernel_transform_weights<3, 4> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(w), filters_out, filters_in, invert);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_weights<5, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(w), filters_out, filters_in, invert);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	template<typename T>
	void launch_input_transform(mlContext_t context, int tile_size, const mlTensor_t x, mlTensor_t matrices)
	{
		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int filters = x.dim[3] / vector_length<T>();

		const int kernel_size = get_kernel_size(matrices, tile_size);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockSize(std::min(128, filters));
		dim3 gridSize(tiles_h, tiles_w, x.dim[0]);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_input<3, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(x), batch_size, height, width, filters);
			if (tile_size == 4)
				kernel_transform_input<3, 4> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(x), batch_size, height, width, filters);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_input<5, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(x), batch_size, height, width, filters);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	template<typename T>
	void launch_output_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, const mlTensor_t bias, const mlTensor_t ext,
			mlTensor_t y, mlActivationType_t act)
	{
		const int batch_size = y.dim[0];
		const int height = y.dim[1];
		const int width = y.dim[2];
		const int filters = y.dim[3] / vector_length<T>();

		const int kernel_size = get_kernel_size(matrices, tile_size);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockSize(std::min(128, filters));
		dim3 gridSize(tiles_h, tiles_w, batch_size);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_output<3, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(y), data<T>(ext), data<T>(bias), act,
						batch_size, height, width, filters);
			if (tile_size == 4)
				kernel_transform_output<3, 4> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(y), data<T>(ext), data<T>(bias), act,
						batch_size, height, width, filters);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_output<5, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(y), data<T>(ext), data<T>(bias), act,
					batch_size, height, width, filters);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	template<typename T>
	void launch_gradient_transform(mlContext_t context, int tile_size, const mlTensor_t dy, mlTensor_t matrices)
	{
		const int batch_size = dy.dim[0];
		const int height = dy.dim[1];
		const int width = dy.dim[2];
		const int filters = dy.dim[3] / vector_length<T>();

		const int kernel_size = get_kernel_size(matrices, tile_size);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockSize(std::min(128, filters));
		dim3 gridSize(tiles_h, tiles_w, dy.dim[0]);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_gradient<3, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(dy), batch_size, height, width,
						filters);
			if (tile_size == 4)
				kernel_transform_gradient<3, 4> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(dy), batch_size, height, width,
						filters);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_gradient<5, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<T>(dy), batch_size, height, width, filters);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	template<typename T, typename U>
	void launch_update_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, mlTensor_t dw)
	{
		const int filters_out = dw.dim[0];
		const int filters_in = dw.dim[3] / vector_length<T>();

		const int kernel_size = get_kernel_size(matrices, tile_size);

		dim3 blockSize(std::min(128, filters_in));
		dim3 gridSize(1, filters_out);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_update<3, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<U>(dw), filters_out, filters_in);
			if (tile_size == 4)
				kernel_transform_update<3, 4> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<U>(dw), filters_out, filters_in);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_update<5, 2> <<<gridSize, blockSize, 0, stream>>>(data<T>(matrices), data<U>(dw), filters_out, filters_in);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	void cuda_winograd_weight_transform(mlContext_t context, int tile_size, const mlTensor_t w, mlTensor_t matrices, bool invert)
	{
		assert(w.dtype == matrices.dtype);
		switch (matrices.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(w) % 2 == 0)
					launch_weight_transform<half2>(context, tile_size, w, matrices, invert);
				else
					launch_weight_transform<half>(context, tile_size, w, matrices, invert);
				break;
			}
			case DTYPE_FLOAT32:
				launch_weight_transform<float>(context, tile_size, w, matrices, invert);
				break;
		}
	}
	void cuda_winograd_input_transform(mlContext_t context, int tile_size, const mlTensor_t x, mlTensor_t matrices)
	{
		assert(x.dtype == matrices.dtype);
		switch (matrices.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(x) % 2 == 0)
					launch_input_transform<half2>(context, tile_size, x, matrices);
				else
					launch_input_transform<half>(context, tile_size, x, matrices);
				break;
			}
			case DTYPE_FLOAT32:
				launch_input_transform<float>(context, tile_size, x, matrices);
				break;
		}
	}
	void cuda_winograd_output_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, const mlTensor_t bias, const mlTensor_t ext,
			mlTensor_t y, mlActivationType_t act)
	{
		assert(matrices.dtype == y.dtype);
		switch (matrices.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(y) % 2 == 0)
					launch_output_transform<half2>(context, tile_size, matrices, bias, ext, y, act);
				else
					launch_output_transform<half>(context, tile_size, matrices, bias, ext, y, act);
				break;
			}
			case DTYPE_FLOAT32:
				launch_output_transform<float>(context, tile_size, matrices, bias, ext, y, act);
				break;
		}
	}
	void cuda_winograd_gradient_transform(mlContext_t context, int tile_size, const mlTensor_t dy, mlTensor_t matrices)
	{
		assert(matrices.dtype == dy.dtype);
		switch (matrices.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (get_last_dim(dy) % 2 == 0)
					launch_gradient_transform<half2>(context, tile_size, dy, matrices);
				else
					launch_gradient_transform<half>(context, tile_size, dy, matrices);
				break;
			}
			case DTYPE_FLOAT32:
				launch_gradient_transform<float>(context, tile_size, dy, matrices);
				break;
		}
	}
	void cuda_winograd_update_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, mlTensor_t dw)
	{
		switch (matrices.dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(is_fp32(dw) || is_fp16(dw));
				if (is_fp16(dw))
					launch_update_transform<half, half>(context, tile_size, matrices, dw);
				else
					launch_update_transform<half, float>(context, tile_size, matrices, dw);
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(is_fp32(dw));
				launch_update_transform<float, float>(context, tile_size, matrices, dw);
				break;
			}
		}
	}

//	void cuda_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
//			void *matrices, bool invert)
//	{
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				if (get_last_dim(weight_shape) % 2 == 0)
//					launch_weight_transform<half2>(context, tile_size, weight_shape, weights, matrices, invert);
//				else
//					launch_weight_transform<half>(context, tile_size, weight_shape, weights, matrices, invert);
//				break;
//			}
//			case DTYPE_FLOAT32:
//				launch_weight_transform<float>(context, tile_size, weight_shape, weights, matrices, invert);
//				break;
//		}
//	}
//	void cuda_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
//			const void *input, void *matrices)
//	{
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				if (get_last_dim(input_shape) % 2 == 0)
//					launch_input_transform<half2>(context, tile_size, weight_shape, input_shape, input, matrices);
//				else
//					launch_input_transform<half>(context, tile_size, weight_shape, input_shape, input, matrices);
//				break;
//			}
//			case DTYPE_FLOAT32:
//				launch_input_transform<float>(context, tile_size, weight_shape, input_shape, input, matrices);
//				break;
//		}
//	}
//	void cuda_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
//			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
//	{
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				if (get_last_dim(output_shape) % 2 == 0)
//					launch_output_transform<half2>(context, tile_size, weight_shape, output_shape, matrices, output, bias, add, act);
//				else
//					launch_output_transform<half>(context, tile_size, weight_shape, output_shape, matrices, output, bias, add, act);
//				break;
//			}
//			case DTYPE_FLOAT32:
//				launch_output_transform<float>(context, tile_size, weight_shape, output_shape, matrices, output, bias, add, act);
//				break;
//		}
//	}
//	void cuda_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
//			const void *gradient, void *matrices)
//	{
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				if (get_last_dim(gradient_shape) % 2 == 0)
//					launch_gradient_transform<half2>(context, tile_size, dtype, weight_shape, gradient_shape, gradient, matrices);
//				else
//					launch_gradient_transform<half>(context, tile_size, dtype, weight_shape, gradient_shape, gradient, matrices);
//				break;
//			}
//			case DTYPE_FLOAT32:
//				launch_gradient_transform<float>(context, tile_size, dtype, weight_shape, gradient_shape, gradient, matrices);
//				break;
//		}
//	}
//	void cuda_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
//			void *update)
//	{
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				if (get_last_dim(weight_shape) % 2 == 0)
//					launch_update_transform<half2>(context, tile_size, dtype, weight_shape, matrices, update);
//				else
//					launch_update_transform<half>(context, tile_size, dtype, weight_shape, matrices, update);
//				break;
//			}
//			case DTYPE_FLOAT32:
//				launch_update_transform<float>(context, tile_size, dtype, weight_shape, matrices, update);
//				break;
//		}
//	}

} /* namespace ml */

