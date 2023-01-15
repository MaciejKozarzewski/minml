/*
 * winograd_fused.cu
 *
 *  Created on: Jan 7, 2023
 *      Author: maciek
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "winograd_transforms.cuh"
#include "../vectors/vectors.cuh"
#include "../utils.hpp"

#include "../helpers/indexers.cuh"
#include "../helpers/tensor_wrappers.cuh"
#include "../helpers/lines_and_tiles.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <iostream>

namespace
{
	using namespace ml;
	using namespace vectors;

	__device__ constexpr int square(int x)
	{
		return x * x;
	}
	__device__ int2 get_number_of_tiles(int height, int width, int tile_size)
	{
		const int h = (height + tile_size - 1) / tile_size;
		const int w = (width + tile_size - 1) / tile_size;
		return int2 { h, w };
	}

	template<typename T>
	__device__ void mul_add(Line<T, 4> &acc, const T &lhs, const Line<T, 4> &rhs)
	{
		acc.x0 += lhs * rhs.x0;
		acc.x1 += lhs * rhs.x1;
		acc.x2 += lhs * rhs.x2;
		acc.x3 += lhs * rhs.x3;
	}
	template<typename T>
	__device__ void mul_add(Line<T, 8> &acc, const T &lhs, const Line<T, 8> &rhs)
	{
		acc.x0 += lhs * rhs.x0;
		acc.x1 += lhs * rhs.x1;
		acc.x2 += lhs * rhs.x2;
		acc.x3 += lhs * rhs.x3;
		acc.x4 += lhs * rhs.x4;
		acc.x5 += lhs * rhs.x5;
		acc.x6 += lhs * rhs.x6;
		acc.x7 += lhs * rhs.x7;
	}

	template<typename T>
	__device__ void outer_product(Tile<T, 4, 4> &acc, const Line<T, 4> &lhs, const Line<T, 4> &rhs)
	{
		mul_add(acc.x0, lhs.x0, rhs);
		mul_add(acc.x1, lhs.x1, rhs);
		mul_add(acc.x2, lhs.x2, rhs);
		mul_add(acc.x3, lhs.x3, rhs);
	}
	template<typename T>
	__device__ void outer_product(Tile<T, 4, 8> &acc, const Line<T, 4> &lhs, const Line<T, 8> &rhs)
	{
		mul_add(acc.x0, lhs.x0, rhs);
		mul_add(acc.x1, lhs.x1, rhs);
		mul_add(acc.x2, lhs.x2, rhs);
		mul_add(acc.x3, lhs.x3, rhs);
	}
	template<typename T>
	__device__ void outer_product(Tile<T, 8, 8> &acc, const Line<T, 8> &lhs, const Line<T, 8> &rhs)
	{
		mul_add(acc.x0, lhs.x0, rhs);
		mul_add(acc.x1, lhs.x1, rhs);
		mul_add(acc.x2, lhs.x2, rhs);
		mul_add(acc.x3, lhs.x3, rhs);
		mul_add(acc.x4, lhs.x4, rhs);
		mul_add(acc.x5, lhs.x5, rhs);
		mul_add(acc.x6, lhs.x6, rhs);
		mul_add(acc.x7, lhs.x7, rhs);
	}

	template<typename T>
	__global__ void kernel_transform_weights(T *matrices, const T *weights, const int input_filters)
	{
//		const int output_filters = gridDim.y;
//		const T c05 = static_cast<T>(0.5f);
//
//		T tile[3][3];
//		for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < input_filters; f += gridDim.x * blockDim.x)
//		{
//			{
//				const Tensor4DIndexer weights_indexer(output_filters, 3, 3, input_filters);
//				for (int i = 0; i < 3; i++)
//					for (int j = 0; j < 3; j++)
//						tile[i][j] = weights[weights_indexer.at(blockIdx.y, i, j, f)];
//			}
//
//			for (int row = 0; row < 4; row++)
//			{
//				T transformed_row[3];
//				switch (row)
//				{
//					case 0:
//						for (int c = 0; c < 3; c++)
//							transformed_row[c] = tile[0][c];
//						break;
//					case 1:
//						for (int c = 0; c < 3; c++)
//							transformed_row[c] = c05 * (tile[0][c] + tile[1][c] + tile[2][c]);
//						break;
//					case 2:
//						for (int c = 0; c < 3; c++)
//							transformed_row[c] = c05 * (tile[0][c] - tile[1][c] + tile[2][c]);
//						break;
//					case 3:
//						for (int c = 0; c < 3; c++)
//							transformed_row[c] = tile[2][c];
//						break;
//				}
//
//				const T tmp0 = transformed_row[0];
//				const T tmp1 = c05 * (transformed_row[0] + transformed_row[1] + transformed_row[2]);
//				const T tmp2 = c05 * (transformed_row[0] - transformed_row[1] + transformed_row[2]);
//				const T tmp3 = transformed_row[2];
//
//				{
//					const Tensor3DIndexer matrices_indexer(16, output_filters, input_filters);
//					matrices[matrices_indexer.at(row * 4 + 0, blockIdx.y, f)] = tmp0;
//					matrices[matrices_indexer.at(row * 4 + 1, blockIdx.y, f)] = tmp1;
//					matrices[matrices_indexer.at(row * 4 + 2, blockIdx.y, f)] = tmp2;
//					matrices[matrices_indexer.at(row * 4 + 3, blockIdx.y, f)] = tmp3;
//				}
//			}
//		}
	}

	__device__ int grid_volume()
	{
		return gridDim.x * gridDim.y * gridDim.z;
	}

	template<int X, int Y, int Z = 1>
	__device__ dim3 reshape_thread_block(int thread_idx)
	{
		assert(X * Y * Z == blockDim.x);
		dim3 result;
		result.x = thread_idx % X;
		result.y = (thread_idx / X) % Y;
		result.z = thread_idx / (X * Y);
		assert(result.x < X);
		assert(result.y < Y);
		assert(result.z < Z);
		return result;
	}

	template<int X, int Y>
	__device__ int2 divide_thread_block()
	{
		assert(X * Y == blockDim.x);
		int2 result;
		result.x = blockIdx.x / X;
		result.y = blockIdx.x % X;
		return result;
	}

//	template<typename T, int KernelSize, int InputSpatialTile, int InputFiltersTile>
//	__device__ void load_input_fragment(ConstTensorWrapper<4, T> &input, int height, int width, int input_filter_idx)
//	{
//		assert(blockDim.x == 256);
//		constexpr int Padding = KernelSize / 2;
//		constexpr int TotalTileSize = InputSpatialTile + 2 * Padding;
//
//		const int2 idx = divide_thread_block<8, 32>();
//
//		for (int row = 0; row < TotalTileSize; row++)
//			for (int col = 0; col < TotalTileSize; col++)
//				if (input_filter_idx < input.last_dim())
//				{
//					const int h = InputSpatialTile * blockIdx.x - Padding + row;
//					const int w = InputSpatialTile * blockIdx.y - Padding + col;
//				}
//	}

	template<typename T>
	__device__ Tile<T, 4, 4> load_input_tile(ConstTensorWrapper<4, T> &input, int height, int width, int batch_idx, int tile_h_idx, int tile_w_idx,
			int input_filter_idx)
	{
		Tile<Vector<T>, 4, 4> result;
		for (int col = 0; col < 4; col++)
			for (int row = 0; row < 4; row++)
			{
				const int h = 2 * tile_h_idx - 1 + row;
				const int w = 2 * tile_w_idx - 1 + col;
				if (0 <= h and h < height and 0 <= w and w < width)
					result.at(col, row) = input.load(batch_idx, h, w, input_filter_idx);
				else
					result.at(col, row) = vector_zero<T>();
			}
		return result;
	}
	template<typename T>
	__device__ void transform_input_tile(TensorWrapper<4, T> &matrices, const Tile<T, 4, 4> &tile, int filter_offset)
	{
		const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
		const Transform<TransformType::INPUT, 3, 2, T> transform;
		for (int row = 0; row < 4; row++)
		{
			Line<Vector<T>, 4> line;
			for (int col = 0; col < 4; col++)
				line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

			for (int col = 0; col < 4; col++)
			{
				const Vector<T> tmp = transform(col, line);
				matrices.store(tmp, row, col, tile_index, filter_offset);
			}
		}
	}

	template<typename T>
	__device__ Tile<Vector<T>, 3, 3> load_weight_tile(ConstTensorWrapper<4, T> &weights, int output_filter_idx, int input_filter_idx)
	{
		Tile<Vector<T>, 3, 3> result;
		for (int col = 0; col < 3; col++)
			for (int row = 0; row < 3; row++)
				result.at(col, row) = weights.load(output_filter_idx, row, col, input_filter_idx);
		return result;
	}
	template<typename T>
	__device__ void transform_weight_tile(TensorWrapper<4, T> &matrices, const Tile<Vector<T>, 3, 3> &tile, int output_filter_idx,
			int input_filter_idx)
	{
		const Transform<TransformType::WEIGHT, 3, 2, Vector<T>> transform;
		for (int row = 0; row < 4; row++)
		{
			Line<Vector<T>, 3> line;
			for (int col = 0; col < 3; col++)
				line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

			for (int col = 0; col < 4; col++)
			{
				const Vector<T> tmp = transform(col, line);
				matrices.store(tmp, row, col, output_filter_idx, input_filter_idx);
			}
		}
	}

	template<typename T>
//	__launch_bounds__(256, 2)
	__global__ void kernel_fused_winograd(const T *input, const T *weights, T *output, int batch_size, int height, int width, int input_filters,
			int output_filters)
	{
		assert(blockDim.x == 256);
		/*
		 * blockIdx.x <- output filters tile (32 or 64)
		 * blockIdx.y <- index of input spatial tile (8x8)
		 * blockIdx.z <- batch index
		 */
		constexpr int KernelSize = 3;
		constexpr int Padding = KernelSize / 2;
		constexpr int TransformSize = 2;
		constexpr int TransformTileSize = TransformSize + KernelSize - 1;

		constexpr int InputSpatialTile = 8;
		constexpr int TotalTileSize = InputSpatialTile + 2 * Padding;

		constexpr int InputFiltersTile = 32 / sizeof(T);
		constexpr int OutputFiltersTile = 32; // maybe check also 64

		__shared__ Vector<T> input_matrices[square(TotalTileSize) * InputFiltersTile];
		__shared__ T weights_matrices[square(TransformTileSize) * OutputFiltersTile * InputFiltersTile];

		const int2 tile_layout = get_number_of_tiles(height, width, TotalTileSize);
		const int tile_index_h = blockIdx.y / tile_layout.x;
		const int tile_index_w = blockIdx.y % tile_layout.x;

		Tile<Vector<T>, 4, 8> accumulator;
		accumulator.fill(vector_zero<T>());

		for (int in = 0; in < input_filters; in += InputFiltersTile)
		{ // loop over dot product length (input filters)

//			if (threadIdx.x < blockDim.x / 2)
//			{ // lower half of threads load input data
//				ConstTensorWrapper<4, T> input_wrapper(input, batch_size, height, width, input_filters);
//				const int h = TotalTileSize * tile_index_h + 2 * (vertical_thread_idx / 4);
//				const int w = TotalTileSize * tile_index_w + 2 * (vertical_thread_idx % 4);
//				const int f = in + horizontal_thread_idx;
//				Tile<Vector<T>, 4, 4> tile = load_input_tile(input_wrapper, height, width, blockIdx.z, h, w, f);
//
//				TensorWrapper<4, T> matrices_wrapper(input_matrices, TransformTileSize, TransformTileSize, grid_volume(), input_filters);
//				transform_input_tile(matrices_wrapper, tile, in);
//			}
//			else
			{
				const dim3 tmp_thread_idx = reshape_thread_block<8, 32>(threadIdx.x);
				ConstTensorWrapper<4, T> weight_wrapper(input, output_filters, KernelSize, KernelSize, input_filters);
				TensorWrapper<4, T> matrices_wrapper(weights_matrices, TransformTileSize, TransformTileSize, OutputFiltersTile, InputFiltersTile);

				for (int f = tmp_thread_idx.y; f < OutputFiltersTile; f += 32)
				{
					const int f_in = in + tmp_thread_idx.x;
					const int f_out = blockIdx.x * OutputFiltersTile + f;
					Tile<Vector<T>, 3, 3> tile = load_weight_tile(weight_wrapper, f_out, f_in);
					transform_weight_tile(matrices_wrapper, tile, f, tmp_thread_idx.x);
				}
			}

			for (int i = threadIdx.x; i < square(TransformTileSize) * OutputFiltersTile * InputFiltersTile; i += blockDim.x)
				output[i] = weights_matrices[i];

//			const Tensor4DIndexer input_indexer(batch_size, height, width, input_filters);
//			for (int in_tile_idx = vertical_thread_idx; in_tile_idx < square(TotalTileSize); in_tile_idx += 16)
//				for (int f = horizontal_thread_idx; f < InputFiltersTile; f += 16)
//				{
//					const int in_tile_h = in_tile_idx / TotalTileSize;
//					const int in_tile_w = in_tile_idx % TotalTileSize;
//					const int h = InputSpatialTile * tile_index_h - Padding + in_tile_h;
//					const int w = InputSpatialTile * tile_index_w - Padding + in_tile_w;
//
//					const int workspace_idx = (in_tile_h * TotalTileSize + in_tile_w) * InputFiltersTile + f;
//					if (0 <= h and h < height and 0 <= w and w < width and (in + f) < input_filters)
//					{
//						const int input_idx = ((blockIdx.z * height + h) * width + w) * input_filters + in + f;
//						input_workspace[workspace_idx] = input[input_idx];
//					}
//					else
//						input_workspace[workspace_idx] = 0.0f;
//				}
//
//			// load weights
//			for (int out_filter_idx = vertical_thread_idx; out_filter_idx < OutputFiltersTile; out_filter_idx += 16)
//				for (int k = 0; k < square(KernelSize); k++) // loop over kernel elements
//					for (int f = horizontal_thread_idx; f < InputFiltersTile; f += 16)
//					{
//						const int workspace_idx = (out_filter_idx * square(KernelSize) + k) * InputFiltersTile + f;
//						if ((in + f) < input_filters and (blockIdx.x * OutputFiltersTile + out_filter_idx) < output_filters)
//						{
//							const int weights_idx = ((blockIdx.x + out_filter_idx) * square(KernelSize) + k) * input_filters + in + f;
//							weights_workspace[workspace_idx] = input[weights_idx];
//						}
//						else
//							weights_workspace[workspace_idx] = 0.0f;
//					}
			__syncthreads();

//			constexpr int HalfTileSize = InputSpatialTile / 2;
//			for (int kernel_h = 0; kernel_h < KernelSize; kernel_h++)
//				for (int kernel_w = 0; kernel_w < KernelSize; kernel_w++)
//					for (int f = 0; f < InputFiltersTile; f++)
//					{
//						int sub_tile_h = vertical_thread_idx / 4;
//						int sub_tile_w = vertical_thread_idx % 4;
//						int offset = ((kernel_h + sub_tile_h) * InputSpatialTile + kernel_w + sub_tile_w) * InputFiltersTile + f;
//						const float inp00 = input_workspace[offset];
//						const float inp01 = input_workspace[offset + HalfTileSize * InputFiltersTile];
//						const float inp10 = input_workspace[offset + HalfTileSize * InputSpatialTile * InputFiltersTile];
//						const float inp11 = input_workspace[offset + HalfTileSize * (1 + InputSpatialTile) * InputFiltersTile];
//
//						// OutputFiltersTile * square(KernelSize) * InputFiltersTile
//						offset = ((horizontal_thread_idx * KernelSize + kernel_h) * KernelSize + kernel_w) * InputFiltersTile + f;
//						const int stride = square(KernelSize) * InputFiltersTile;
//						const float w0 = weights_workspace[offset + 0 * stride];
//						const float w1 = 1.0f; // weights_workspace[offset + 1 * stride];
//						const float w2 = 1.0f; // weights_workspace[offset + 2 * stride];
//						const float w3 = 1.0f; // weights_workspace[offset + 3 * stride];
//
//						output_tile[0][0] += inp00 * w0;
//						output_tile[0][1] += inp00 * w1;
//						output_tile[0][2] += inp00 * w2;
//						output_tile[0][3] += inp00 * w3;
//
//						output_tile[1][0] += inp01 * w0;
//						output_tile[1][1] += inp01 * w1;
//						output_tile[1][2] += inp01 * w2;
//						output_tile[1][3] += inp01 * w3;
//
//						output_tile[2][0] += inp10 * w0;
//						output_tile[2][1] += inp10 * w1;
//						output_tile[2][2] += inp10 * w2;
//						output_tile[2][3] += inp10 * w3;
//
//						output_tile[3][0] += inp11 * w0;
//						output_tile[3][1] += inp11 * w1;
//						output_tile[3][2] += inp11 * w2;
//						output_tile[3][3] += inp11 * w3;
//					}
			__syncthreads();
		}

//		const int sub_tile_h = vertical_thread_idx / 4;
//		const int sub_tile_w = vertical_thread_idx % 4;
//		for (int i = 0; i < 4; i++)
//		{
//			for (int k = 0; k < 4; k++)
//				input_workspace[(sub_tile_h * 4 + sub_tile_w) * OutputFiltersTile + k * 16 + horizontal_thread_idx] = output_tile[i][k];
//
//			const int store_tile_h = threadIdx.x / 64;
//			const int store_tile_w = threadIdx.x % 64;
//
//			const int in_tile_h = (i / 2) * 4 + sub_tile_h;
//			const int in_tile_w = (i % 2) * 4 + sub_tile_w;
//			const int h = InputSpatialTile * tile_index_h + in_tile_h;
//			const int w = InputSpatialTile * tile_index_w + in_tile_w;
//
//			const int output_filter_idx = blockIdx.x * OutputFiltersTile + store_tile_w;
////			if (0 <= h and h < height and 0 <= w and w < width)// and output_filter_idx < output_filters)
//			{
//				const int output_idx = ((blockIdx.z * height + h) * width + w) * output_filters + output_filter_idx;
//				output[output_idx] = input_workspace[store_tile_h * 64 + store_tile_w];
//			}
//		}
	}

	int get_output_filters_tile(int kernel_size)
	{
		switch (kernel_size)
		{
			case 1:
				return 128;
			case 3:
				return 64;
			case 5:
				return 32;
		}
		return 0;
	}
}

namespace ml
{
	void cuda_convolution_fused_winograd_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int input_filters = input_shape.dim[3];
		const int output_filters = weights_shape.dim[0];

		const int filters_tile = 32; //get_output_filters_tile(weight_shape.dim[1]);

		const int tiles_h = (height + 7) / 8;
		const int tiles_w = (width + 7) / 8;
		const int tiles_out = (output_filters + filters_tile - 1) / filters_tile;
		cudaStream_t stream = cuda::Context::getStream(context);

//		dim3 blockSize(128);
//		dim3 gridSize((input_filters + blockSize.x - 1) / blockSize.x, output_filters);
//		kernel_transform_weights <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(workspace), getPointer<float>(weights), input_filters);

		dim3 blockSize(256);
		dim3 gridSize(tiles_out, tiles_h * tiles_w, batch_size);
		kernel_fused_winograd <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(input), getPointer<float>(weights), getPointer<float>(output),
				batch_size, height, width, input_filters, output_filters);

		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

