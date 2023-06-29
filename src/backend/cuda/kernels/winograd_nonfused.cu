/*
 * winograd.cu
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
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
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <iostream>

namespace
{
	using namespace ml;
	using namespace vectors;

	template<typename T>
	__device__ T fake_low_precision(T x)
	{
		return x;
	}
	__device__ Vector<float> fake_low_precision(Vector<float> x)
	{
		uint32_t tmp = reinterpret_cast<uint32_t*>(&x)[0] & 0xFFFFF000u;
		return reinterpret_cast<float*>(&tmp)[0];
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_weights(T *__restrict__ matrices, const T *__restrict__ weights, int output_filters, int input_filters,
			bool invert, bool low_precision)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<Vector<T>, KernelSize, KernelSize> tile;
		for (int f = threadIdx.x * vector_length<T>(); f < input_filters; f += blockDim.x * vector_length<T>())
		{
			ConstTensorWrapper<4, T> weights_wrapper(weights, output_filters, KernelSize, KernelSize, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					Vector<T> tmp;
					if (invert)
						tmp = weights_wrapper.load(blockIdx.y, KernelSize - 1 - row, KernelSize - 1 - col, f);
					else
						tmp = weights_wrapper.load(blockIdx.y, row, col, f);
					if (low_precision)
						tmp = fake_low_precision(tmp);
					tile.at(col, row) = tmp;
				}

			TensorWrapper<4, T> matrices_wrapper(matrices, TileSize, TileSize, output_filters, input_filters);

			const Transform<TransformType::WEIGHT, KernelSize, TransformSize, Vector<T>> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<Vector<T>, KernelSize> line;
				for (int col = 0; col < KernelSize; col++)
					line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

				for (int col = 0; col < TileSize; col++)
				{
					const Vector<T> tmp = transform(col, line);
					matrices_wrapper.store(tmp, row, col, blockIdx.y, f);
				}
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_input(T *__restrict__ matrices, const T *__restrict__ input, int batch_size, int height, int width,
			int input_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;
		constexpr int Padding = KernelSize / 2;

		Tile<Vector<T>, TileSize, TileSize> tile;
		for (int f = threadIdx.x * vector_length<T>(); f < input_filters; f += blockDim.x * vector_length<T>())
		{
			ConstTensorWrapper<4, T> input_wrapper(input, batch_size, height, width, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					const int h = TransformSize * blockIdx.x - Padding + row;
					const int w = TransformSize * blockIdx.y - Padding + col;
					if (0 <= h && h < height && 0 <= w && w < width)
						tile.at(col, row) = input_wrapper.load(blockIdx.z, h, w, f);
					else
						tile.at(col, row) = vector_zero<T>();
				}

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			TensorWrapper<4, T> matrices_wrapper(matrices, TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, input_filters);
			const Transform<TransformType::INPUT, KernelSize, TransformSize, Vector<T>> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<Vector<T>, TileSize> line;
				for (int col = 0; col < TileSize; col++)
					line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

				for (int col = 0; col < TileSize; col++)
				{
					const Vector<T> tmp = transform(col, line);
					matrices_wrapper.store(tmp, row, col, tile_index, f);
				}
			}
		}
	}
	template<int KernelSize, int TransformSize>
	__global__ void kernel_transform_input_vect(half *__restrict__ matrices, const half *__restrict__ input, int batch_size, int height, int width,
			int input_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;
		constexpr int Padding = KernelSize / 2;
		assert(input_filters % 2 == 0);
		input_filters /= 2;

		Tile<Vector<half>, TileSize, TileSize> tile;
		for (int f = threadIdx.x; f < input_filters; f += blockDim.x)
		{
			Indexer<4> input_indexer(batch_size, height, width, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					const int h = TransformSize * blockIdx.x - Padding + row;
					const int w = TransformSize * blockIdx.y - Padding + col;
					if (0 <= h && h < height && 0 <= w && w < width)
						tile.at(col, row) = reinterpret_cast<const half2*>(input)[input_indexer.at(blockIdx.z, h, w, f)];
					else
						tile.at(col, row) = vector_zero<half>();
				}

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			Indexer<4> matrices_indexer(TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, input_filters);
			const Transform<TransformType::INPUT, KernelSize, TransformSize, Vector<half>> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<Vector<half>, TileSize> line;
				for (int col = 0; col < TileSize; col++)
					line[col] = transform(row, tile.get_row(col)); // tile is stored as transposed (column-major)

				for (int col = 0; col < TileSize; col++)
				{
					const Vector<half> tmp = transform(col, line);
					reinterpret_cast<half2*>(matrices)[matrices_indexer.at(row, col, tile_index, f)] = static_cast<half2>(tmp);
				}
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_output(const T *__restrict__ matrices, T *__restrict__ output, const T *__restrict__ add,
			const T *__restrict__ bias, mlActivationType_t activation, int batch_size, int height, int width, int output_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<Vector<T>, TileSize, TileSize> tile;
		for (int f = threadIdx.x * vector_length<T>(); f < output_filters; f += blockDim.x * vector_length<T>())
		{
			const Vector<T> bias_value = (bias != nullptr) ? load_vector(bias, Indexer<1>(output_filters), f) : vector_zero<T>();

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			ConstTensorWrapper<4, T> matrices_wrapper(matrices, TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, output_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
					tile.at(col, row) = matrices_wrapper.load(row, col, tile_index, f);

			TensorWrapper<4, T> output_wrapper(output, batch_size, height, width, output_filters);
			const Transform<TransformType::OUTPUT, KernelSize, TransformSize, Vector<T>> transform;
			for (int row = 0; row < TransformSize; row++)
			{
				const int h = TransformSize * blockIdx.x + row;
				if (h < height)
				{
					Line<Vector<T>, TileSize> line;
					for (int col = 0; col < TileSize; col++)
						line[col] = transform(row, tile.get_row(col));

					for (int col = 0; col < TransformSize; col++)
					{
						const int w = TransformSize * blockIdx.y + col;
						if (w < width)
						{
							Vector<T> tmp = transform(col, line);

							if (add != nullptr)
								tmp += load_vector(add, output_wrapper.indexer, blockIdx.z, h, w, f);
							if (bias != nullptr)
								tmp += bias_value;
							if (activation == ACTIVATION_RELU)
								tmp = vectors::max(vector_zero<T>(), tmp);
							if (activation == ACTIVATION_TANH)
								tmp = vectors::tanh(tmp);
							if (activation == ACTIVATION_SIGMOID)
								tmp = vector_one<T>() / (vector_one<T>() + vectors::exp(-tmp));

							output_wrapper.store(tmp, blockIdx.z, h, w, f);
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

		Tile<Vector<T>, TransformSize, TransformSize> tile;
		for (int f = threadIdx.x * vector_length<T>(); f < output_filters; f += blockDim.x * vector_length<T>())
		{
			ConstTensorWrapper<4, T> gradient_wrapper(gradient, batch_size, height, width, output_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
				{
					const int h = TransformSize * blockIdx.x + row;
					const int w = TransformSize * blockIdx.y + col;
					if (0 <= h && h < height && 0 <= w && w < width)
						tile.at(col, row) = gradient_wrapper.load(blockIdx.z, h, w, f);
					else
						tile.at(col, row) = vector_zero<T>();
				}

			const int tile_index = (blockIdx.z * gridDim.x + blockIdx.x) * gridDim.y + blockIdx.y;
			TensorWrapper<4, T> matrices_wrapper(matrices, TileSize, TileSize, gridDim.x * gridDim.y * gridDim.z, output_filters);
			const Transform<TransformType::GRADIENT, KernelSize, TransformSize, Vector<T>> transform;
			for (int row = 0; row < TileSize; row++)
			{
				Line<Vector<T>, TransformSize> line;
				for (int col = 0; col < TransformSize; col++)
					line[col] = transform(row, tile.get_row(col));

				for (int col = 0; col < TileSize; col++)
				{
					const Vector<T> tmp = transform(col, line);
					matrices_wrapper.store(tmp, row, col, tile_index, f);
				}
			}
		}
	}

	template<int KernelSize, int TransformSize, typename T>
	__global__ void kernel_transform_update(const T *__restrict__ matrices, T *__restrict__ update, int output_filters, int input_filters)
	{
		constexpr int TileSize = KernelSize + TransformSize - 1;

		Tile<Vector<T>, TileSize, TileSize> tile;
		for (int f = threadIdx.x * vector_length<T>(); f < input_filters; f += blockDim.x * vector_length<T>())
		{
			ConstTensorWrapper<4, T> matrices_wrapper(matrices, TileSize, TileSize, output_filters, input_filters);
			for (int col = 0; col < tile.columns(); col++)
				for (int row = 0; row < tile.rows(); row++)
					tile.at(col, row) = matrices_wrapper.load(row, col, blockIdx.y, f);

			TensorWrapper<4, T> output_wrapper(update, output_filters, KernelSize, KernelSize, input_filters);
			const Transform<TransformType::UPDATE, KernelSize, TransformSize, Vector<T>> transform;
			for (int row = 0; row < KernelSize; row++)
			{
				Line<Vector<T>, TileSize> line;
				for (int col = 0; col < TileSize; col++)
					line[col] = transform(row, tile.get_row(col));

				for (int col = 0; col < KernelSize; col++)
				{
					const Vector<T> tmp = transform(col, line);
					output_wrapper.store(tmp, blockIdx.y, row, col, f);
				}
			}
		}
	}

	/*
	 * host code
	 */
	template<typename T>
	bool is_fp32()
	{
		return std::is_same<T, float>::value;
	}
	int get_kernel_size(const mlShape_t &weight_shape)
	{
		assert(weight_shape.rank == 4);
		assert(weight_shape.dim[1] == weight_shape.dim[2]);
		return weight_shape.dim[1];
	}
	int get_number_of_tiles(int dim, int transform_size)
	{
		return (dim + transform_size - 1) / transform_size;
	}

	template<typename T>
	void launch_weight_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, const void *weights, void *matrices, bool invert,
			bool low_precision)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
		dim3 blockSize(std::min(max_threads, filters_in));
		dim3 gridSize(1, filters_out);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_weights<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(weights), filters_out,
						filters_in, invert, low_precision);
			if (tile_size == 4)
				kernel_transform_weights<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(weights), filters_out,
						filters_in, invert, low_precision);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_weights<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(weights), filters_out,
					filters_in, invert, low_precision);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	template<typename T>
	void launch_input_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, mlShape_t input_shape, const void *input, void *matrices)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filters = input_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
		dim3 blockSize(std::min(max_threads, filters));
		dim3 gridSize(tiles_h, tiles_w, input_shape.dim[0]);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_input<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(input), batch_size, height,
						width, filters);
			if (tile_size == 4)
			{
				if (filters % 2 == 0 && cuda::has_fp16_math(context) && std::is_same<T, half>::value)
					kernel_transform_input_vect<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<half>(matrices), getPointer<half>(input),
							batch_size, height, width, filters);
				else
					kernel_transform_input<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(input), batch_size,
							height, width, filters);
			}
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			if (filters % 2 == 0 && cuda::has_fp16_math(context) && std::is_same<T, half>::value)
				kernel_transform_input_vect<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<half>(matrices), getPointer<half>(input),
						batch_size, height, width, filters);
			else
				kernel_transform_input<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(input), batch_size, height,
						width, filters);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}

	template<typename T>
	void launch_output_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, mlShape_t output_shape, const void *matrices,
			void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const int batch_size = output_shape.dim[0];
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int filters = output_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
		dim3 blockSize(std::min(max_threads, filters));
		dim3 gridSize(tiles_h, tiles_w, batch_size);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_output<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(output), getPointer<T>(add),
						getPointer<T>(bias), act, batch_size, height, width, filters);
			if (tile_size == 4)
				kernel_transform_output<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(output), getPointer<T>(add),
						getPointer<T>(bias), act, batch_size, height, width, filters);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_output<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<T>(matrices), getPointer<T>(output), getPointer<T>(add),
					getPointer<T>(bias), act, batch_size, height, width, filters);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}

}

namespace ml
{
	void cuda_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
			void *matrices, bool invert, bool low_precision)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				launch_weight_transform<half>(context, tile_size, weight_shape, weights, matrices, invert, low_precision);
				break;
			case DTYPE_FLOAT32:
				launch_weight_transform<float>(context, tile_size, weight_shape, weights, matrices, invert, low_precision);
				break;
		}
	}
	void cuda_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				launch_input_transform<half>(context, tile_size, weight_shape, input_shape, input, matrices);
				break;
			case DTYPE_FLOAT32:
				launch_input_transform<float>(context, tile_size, weight_shape, input_shape, input, matrices);
				break;
		}
	}
	void cuda_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
				launch_output_transform<half>(context, tile_size, weight_shape, output_shape, matrices, output, bias, add, act);
				break;
			case DTYPE_FLOAT32:
				launch_output_transform<float>(context, tile_size, weight_shape, output_shape, matrices, output, bias, add, act);
				break;
		}
	}
	void cuda_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
			const void *gradient, void *matrices)
	{
		const int batch_size = gradient_shape.dim[0];
		const int height = gradient_shape.dim[1];
		const int width = gradient_shape.dim[2];
		const int filters = gradient_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		cudaStream_t stream = cuda::Context::getStream(context);

		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
		dim3 blockSize(std::min(max_threads, filters));
		dim3 gridSize(tiles_h, tiles_w, gradient_shape.dim[0]);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_gradient<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient),
						batch_size, height, width, filters);
			if (tile_size == 4)
				kernel_transform_gradient<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient),
						batch_size, height, width, filters);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_gradient<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient), batch_size,
					height, width, filters);
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
			void *update)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
		dim3 blockSize(std::min(max_threads, filters_in));
		dim3 gridSize(1, filters_out);
		cudaStream_t stream = cuda::Context::getStream(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_update<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update),
						filters_out, filters_in);
			if (tile_size == 4)
				kernel_transform_update<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update),
						filters_out, filters_in);
		}
		if (kernel_size == 5 && tile_size == 2)
		{
			kernel_transform_update<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update), filters_out,
					filters_in);
		}

		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

