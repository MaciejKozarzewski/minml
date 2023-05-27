/*
 * conv2d_winograd_nonfused.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "activations.hpp"
#include "winograd_transforms.hpp"
#include "../helpers/indexers.hpp"
#include "../vectors/vectors.hpp"
#include "../utils.hpp"
#include "../ComputeConfig.hpp"

//#include <omp.h>

namespace
{
	using namespace ml;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	T fake_low_precision(T x)
	{
		return x;
	}
//	Vector<float> emulate_low_precision(Vector<float> x)
//	{
//		return x & Vector<float>(constant<0xFFFFF000u>());
//	}

	template<typename DT, typename CT, int KernelSize, int TileSize>
	void kernel_transform_weights(void *__restrict__ matrices, const void *__restrict__ weights, int output_filters, int input_filters, bool invert,
			bool low_precision)
	{
		constexpr int TransformSize = TileSize + KernelSize - 1;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, output_filters, input_filters);
		const Indexer<4> weights_indexer(output_filters, KernelSize, KernelSize, input_filters);

		const DT *weights_ptr = getPointer<DT>(weights);
		DT *matrices_ptr = getPointer<DT>(matrices);
//#pragma omp parallel
		{
			const DT *ptr_in[KernelSize * KernelSize];
			DT *ptr_out[TransformSize * TransformSize];
			Vector<CT> storage[TransformSize * KernelSize];
//#pragma omp for
			for (int out = 0; out < output_filters; out++)
			{
				for (int i = 0; i < KernelSize; i++)
					for (int j = 0; j < KernelSize; j++)
					{
						int tmp = i * KernelSize + j;
						if (invert)
							ptr_in[tmp] = weights_ptr + weights_indexer.at(out, KernelSize - 1 - i, KernelSize - 1 - j, 0);
						else
							ptr_in[tmp] = weights_ptr + weights_indexer.at(out, i, j, 0);
					}
				for (int k = 0; k < TransformSize * TransformSize; k++)
					ptr_out[k] = matrices_ptr + matrices_indexer.at(k, out, 0);

				Transform<TransformType::WEIGHT, KernelSize, TileSize, CT> transform;
				for (int in = 0; in < input_filters; in += Vector<CT>::size())
				{
					const int elements_left = std::min(input_filters - in, Vector<CT>::size());
					for (int col = 0; col < KernelSize; col++)
					{
						Line<KernelSize, CT> column;
//						if (low_precision)
//						{
//							for (int i = 0; i < column.length(); i++)
//								column[i] = emulate_low_precision(column[i]);
//						}
						column.load_column(ptr_in, col, in, elements_left, KernelSize);
						Line<TransformSize, CT> transformed = transform(column);
						transformed.store_column(storage, col, KernelSize);
					}

					for (int col = 0; col < TransformSize; col++)
					{
						Line<KernelSize, CT> column;
						column.load_row(storage, col, KernelSize);
						Line<TransformSize, CT> transformed = transform(column);
						transformed.store_row(ptr_out, col, in, elements_left, TransformSize);
					}
				}
			}
		}
	}
	template<typename DT, typename CT, int KernelSize, int TileSize>
	void kernel_transform_input(void *__restrict__ matrices, const void *__restrict__ input, int batch_size, int height, int width, int input_filters,
			void *__restrict__ workspace)
	{
		constexpr int Padding = KernelSize / 2;
		constexpr int TransformSize = TileSize + KernelSize - 1;

		const int tiles_h = (height + TileSize - 1) / TileSize;
		const int tiles_w = (width + TileSize - 1) / TileSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, input_filters);
		const Indexer<4> input_indexer(batch_size, height, width, input_filters);

		const DT *input_ptr = getPointer<DT>(input);
		DT *matrices_ptr = getPointer<DT>(matrices);

		DT *zero_line = getPointer<DT>(workspace);
		std::memset(zero_line, 0, sizeof(DT) * input_filters);

//#pragma omp parallel
		{
			const DT *ptr_in[TransformSize * TransformSize];
			DT *ptr_out[TransformSize * TransformSize];
			Vector<CT> storage[TransformSize * TransformSize];
//#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				int batch = tile_idx / tiles_per_image;
				int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				int matrix_idx = 0;
				for (int i = 0; i < TransformSize; i++)
					for (int j = 0; j < TransformSize; j++, matrix_idx++)
					{
						const int x = TileSize * tile_x + i - Padding;
						const int y = TileSize * tile_y + j - Padding;
						ptr_out[matrix_idx] = matrices_ptr + matrices_indexer.at(matrix_idx, tile_idx, 0);
						if (x >= 0 and x < height and y >= 0 and y < width)
							ptr_in[matrix_idx] = input_ptr + input_indexer.at(batch, x, y, 0);
						else
							ptr_in[matrix_idx] = zero_line;
					}

				Transform<TransformType::INPUT, KernelSize, TileSize, CT> transform;
				for (int in = 0; in < input_filters; in += Vector<CT>::size())
				{
					const int elements_left = std::min(input_filters - in, Vector<CT>::size());
					for (int col = 0; col < TransformSize; col++)
					{
						Line<TransformSize, CT> column;
						column.load_column(ptr_in, col, in, elements_left, TransformSize);
						Line<TransformSize, CT> transformed = transform(column);
						transformed.store_column(storage, col, TransformSize);
					}

					for (int col = 0; col < TransformSize; col++)
					{
						Line<TransformSize, CT> column;
						column.load_row(storage, col, TransformSize);
						Line<TransformSize, CT> transformed = transform(column);
						transformed.store_row(ptr_out, col, in, elements_left, TransformSize);
					}
				}
			}
		}
	}
	template<typename DT, typename CT, int KernelSize, int TileSize>
	void kernel_transform_output(const void *__restrict__ matrices, void *__restrict__ output, const void *__restrict__ add,
			const void *__restrict__ bias, mlActivationType_t activation, int batch_size, int height, int width, int output_filters,
			void *__restrict__ workspace)
	{
		constexpr int TransformSize = TileSize + KernelSize - 1;

		const int tiles_h = (height + TileSize - 1) / TileSize;
		const int tiles_w = (width + TileSize - 1) / TileSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;

		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, output_filters);
		const Indexer<4> output_indexer(batch_size, height, width, output_filters);

		DT *zero_line = getPointer<DT>(workspace);
		std::memset(zero_line, 0, sizeof(DT) * output_filters);

//#pragma omp parallel
		{
//			DT *fake_storage = getPointer<DT>(workspace) + (1 + omp_get_thread_num()) * output_filters;
			DT *fake_storage = getPointer<DT>(workspace) + (1 + 0) * output_filters;

			Vector<CT> storage[TileSize * TransformSize];
			const DT *ptr_in[TransformSize * TransformSize];
			DT *ptr_out[TileSize * TileSize];
			const DT *ptr_add[TileSize * TileSize];

//#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				for (int j = 0; j < TransformSize * TransformSize; j++)
					ptr_in[j] = getPointer<DT>(matrices) + matrices_indexer.at(j, tile_idx, 0);

				int batch = tile_idx / tiles_per_image;
				int tile_x = (tile_idx % tiles_per_image) / tiles_w;
				int tile_y = (tile_idx % tiles_per_image) % tiles_w;
				int output_idx = 0;
				for (int i = 0; i < TileSize; i++)
					for (int j = 0; j < TileSize; j++, output_idx++)
					{
						int x = tile_x * TileSize + i;
						int y = tile_y * TileSize + j;
						if (x < height and y < width)
							ptr_out[output_idx] = getPointer<DT>(output) + output_indexer.at(batch, x, y, 0);
						else
							ptr_out[output_idx] = fake_storage;
					}

				if (add != nullptr)
				{
					output_idx = 0;
					for (int i = 0; i < TileSize; i++)
						for (int j = 0; j < TileSize; j++, output_idx++)
						{
							int x = tile_x * TileSize + i;
							int y = tile_y * TileSize + j;
							if (x < height and y < width)
								ptr_add[output_idx] = getPointer<DT>(add) + output_indexer.at(batch, x, y, 0);
							else
								ptr_add[output_idx] = zero_line;
						}
				}

				Transform<TransformType::OUTPUT, KernelSize, TileSize, CT> transform;
				for (int out = 0; out < output_filters; out += Vector<CT>::size())
				{
					const int elements_left = std::min(output_filters - out, Vector<CT>::size());
					for (int col = 0; col < TransformSize; col++)
					{
						Line<TransformSize, CT> column;
						column.load_column(ptr_in, col, out, elements_left, TransformSize);
						Line<TileSize, CT> transformed = transform(column);
						transformed.store_column(storage, col, TransformSize);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<TransformSize, CT> column;
						column.load_row(storage, col, TransformSize);
						Line<TileSize, CT> transformed = transform(column);

						if (bias != nullptr)
						{
							const Vector<CT> tmp(getPointer<DT>(bias) + out, elements_left);
							for (int i = 0; i < transformed.length(); i++)
								transformed[i] += tmp;
						}

						if (add != nullptr)
						{
							Line<TileSize, CT> z_line;
							z_line.load_row(ptr_add, col, out, elements_left, TileSize);
							for (int i = 0; i < transformed.length(); i++)
								transformed[i] += z_line[i];
						}
						switch (activation)
						{
							default:
							case ACTIVATION_LINEAR:
								break;
							case ACTIVATION_SIGMOID:
								for (int i = 0; i < transformed.length(); i++)
									transformed[i] = activation_forward(ACTIVATION_SIGMOID, transformed[i]);
								break;
							case ACTIVATION_TANH:
								for (int i = 0; i < transformed.length(); i++)
									transformed[i] = activation_forward(ACTIVATION_TANH, transformed[i]);
								break;
							case ACTIVATION_RELU:
								for (int i = 0; i < transformed.length(); i++)
									transformed[i] = activation_forward(ACTIVATION_RELU, transformed[i]);
								break;
						}

						transformed.store_row(ptr_out, col, out, elements_left, TileSize);
					}
				}
			}
		}
	}
	template<typename T, int KernelSize, int TileSize>
	void kernel_transform_gradient(T *__restrict__ matrices, const T *__restrict__ gradient, int batch_size, int height, int width,
			int output_filters, T *__restrict__ workspace)
	{
		constexpr int TransformSize = TileSize + KernelSize - 1;

		const int tiles_h = (height + TileSize - 1) / TileSize;
		const int tiles_w = (width + TileSize - 1) / TileSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;

		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, output_filters);
		const Indexer<4> gradient_indexer(batch_size, height, width, output_filters);

		T *zero_line = workspace;
		std::memset(zero_line, 0, sizeof(T) * output_filters);

//#pragma omp parallel
		{
			const T *ptr_in[TransformSize * TransformSize];
			T *ptr_out[TransformSize * TransformSize];
			Vector<T> storage[TransformSize * TileSize];
//#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				int batch = tile_idx / tiles_per_image;
				int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				int matrix_idx = 0;
				for (int i = 0; i < TileSize; i++)
					for (int j = 0; j < TileSize; j++, matrix_idx++)
					{
						int x = TileSize * tile_x + i;
						int y = TileSize * tile_y + j;
						if (x < height and y < width)
							ptr_in[matrix_idx] = gradient + gradient_indexer.at(batch, x, y, 0);
						else
							ptr_in[matrix_idx] = zero_line;
					}
				for (int k = 0; k < TransformSize * TransformSize; k++)
					ptr_out[k] = matrices + matrices_indexer.at(k, tile_idx, 0);

				Transform<TransformType::GRADIENT, KernelSize, TileSize, T> transform;
				for (int out = 0; out < output_filters; out += Vector<T>::size())
				{
					const int elements_left = std::min(output_filters - out, Vector<T>::size());
					for (int col = 0; col < TileSize; col++)
					{
						Line<TileSize, T> column;
						column.load_column(ptr_in, col, out, elements_left, TileSize);
						Line<TransformSize, T> transformed = transform(column);
						transformed.store_column(storage, col, TileSize);
					}

					for (int col = 0; col < TransformSize; col++)
					{
						Line<TileSize, T> column;
						column.load_row(storage, col, TileSize);
						Line<TransformSize, T> transformed = transform(column);
						transformed.store_row(ptr_out, col, out, elements_left, TransformSize);
					}
				}
			}
		}
	}
	template<typename T, int KernelSize, int TileSize>
	void kernel_transform_update(const T *__restrict__ matrices, T *__restrict__ update, int output_filters, int input_filters)
	{
		constexpr int TransformSize = TileSize + KernelSize - 1;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, output_filters, input_filters);
		const Indexer<4> update_indexer(output_filters, KernelSize, KernelSize, input_filters);

//#pragma omp parallel
		{
			const T *ptr_in[TransformSize * TransformSize];
			T *ptr_out[KernelSize * KernelSize];
			Vector<T> storage[TransformSize * KernelSize];
//#pragma omp for
			for (int out = 0; out < output_filters; out++)
			{
				int matrix_index = 0;
				for (int i = 0; i < KernelSize; i++)
					for (int j = 0; j < KernelSize; j++, matrix_index++)
						ptr_out[matrix_index] = update + update_indexer.at(out, i, j, 0);

				for (int k = 0; k < TransformSize * TransformSize; k++)
					ptr_in[k] = matrices + matrices_indexer.at(k, out, 0);

				Transform<TransformType::UPDATE, KernelSize, TileSize, T> transform;
				for (int in = 0; in < input_filters; in += Vector<T>::size())
				{
					const int elements_left = std::min(input_filters - in, Vector<T>::size());
					for (int col = 0; col < TransformSize; col++)
					{
						Line<TransformSize, T> column;
						column.load_column(ptr_in, col, in, elements_left, TransformSize);
						Line<KernelSize, T> transformed = transform(column);
						transformed.store_column(storage, col, TransformSize);
					}

					for (int col = 0; col < KernelSize; col++)
					{
						Line<TransformSize, T> column;
						column.load_row(storage, col, TransformSize);
						const Line<KernelSize, T> transformed = transform(column);
						transformed.store_row(ptr_out, col, in, elements_left, KernelSize);
					}
				}
			}
		}
	}

	int get_kernel_size(const mlShape_t &weight_shape)
	{
		assert(weight_shape.rank == 4);
		assert(weight_shape.dim[1] == weight_shape.dim[2]);
		return weight_shape.dim[1];
	}
	int get_transform_size(const mlShape_t &weight_shape)
	{
		switch (get_kernel_size(weight_shape))
		{
			case 3:
				return 4; // get_last_dim(weight_shape) <= 4 ? 2 : 4;
			case 5:
				return 2;
			default:
				return 0;
		}
	}

	template<typename DT, typename CT>
	void launch_weight_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, const void *weights, void *matrices, bool invert,
			bool low_precision)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_weights<DT, CT, 3, 2>(matrices, weights, filters_out, filters_in, invert, low_precision);
			if (tile_size == 4)
				kernel_transform_weights<DT, CT, 3, 4>(matrices, weights, filters_out, filters_in, invert, low_precision);
			if (tile_size == 5)
				kernel_transform_weights<DT, CT, 3, 5>(matrices, weights, filters_out, filters_in, invert, low_precision);
		}
		if (kernel_size == 5)
		{
			if (tile_size == 2)
				kernel_transform_weights<DT, CT, 5, 2>(matrices, weights, filters_out, filters_in, invert, low_precision);
		}
	}
	template<typename DT, typename CT>
	void launch_input_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, mlShape_t input_shape, const void *input, void *matrices)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filters = input_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		void *workspace = cpu::Context::getWorkspace(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_input<DT, CT, 3, 2>(matrices, input, batch_size, height, width, filters, workspace);
			if (tile_size == 4)
				kernel_transform_input<DT, CT, 3, 4>(matrices, input, batch_size, height, width, filters, workspace);
			if (tile_size == 5)
				kernel_transform_input<DT, CT, 3, 5>(matrices, input, batch_size, height, width, filters, workspace);
		}
		if (kernel_size == 5)
		{
			if (tile_size == 2)
				kernel_transform_input<DT, CT, 5, 2>(matrices, input, batch_size, height, width, filters, workspace);
		}
	}
	template<typename DT, typename CT>
	void launch_output_transform(mlContext_t context, int tile_size, mlShape_t weight_shape, mlShape_t output_shape, const void *matrices,
			void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const int batch_size = output_shape.dim[0];
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int filters = output_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		void *workspace = cpu::Context::getWorkspace(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_output<DT, CT, 3, 2>(matrices, output, add, bias, act, batch_size, height, width, filters, workspace);
			if (tile_size == 4)
				kernel_transform_output<DT, CT, 3, 4>(matrices, output, add, bias, act, batch_size, height, width, filters, workspace);
			if (tile_size == 5)
				kernel_transform_output<DT, CT, 3, 5>(matrices, output, add, bias, act, batch_size, height, width, filters, workspace);
		}
		if (kernel_size == 5)
		{
			if (tile_size == 2)
				kernel_transform_output<DT, CT, 5, 2>(matrices, output, add, bias, act, batch_size, height, width, filters, workspace);
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
			void *matrices, bool invert, bool low_precision)
	{
		const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
		CREATE_KERNEL_TABLE(launch_weight_transform);
		CALL_KERNEL(launch_weight_transform, cfg)(context, tile_size,weight_shape, weights, matrices, invert, low_precision);
	}
	void cpu_kernel_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
		CREATE_KERNEL_TABLE(launch_input_transform);
		CALL_KERNEL(launch_input_transform, cfg)(context, tile_size,weight_shape, input_shape, input, matrices);
	}
	void cpu_kernel_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
		CREATE_KERNEL_TABLE(launch_output_transform);
		CALL_KERNEL(launch_output_transform, cfg)(context, tile_size,weight_shape, output_shape, matrices, output, bias, add, act);
	}
	void cpu_kernel_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
			mlShape_t gradient_shape, const void *gradient, void *matrices)
	{
		const int batch_size = gradient_shape.dim[0];
		const int height = gradient_shape.dim[1];
		const int width = gradient_shape.dim[2];
		const int filters = gradient_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		float *workspace = cpu::Context::getWorkspace<float>(context);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_gradient<float, 3, 2>(getPointer<float>(matrices), getPointer<float>(gradient), batch_size, height, width, filters,
						workspace);
			if (tile_size == 4)
				kernel_transform_gradient<float, 3, 4>(getPointer<float>(matrices), getPointer<float>(gradient), batch_size, height, width, filters,
						workspace);
		}
		if (kernel_size == 5)
		{
			if (tile_size == 2)
				kernel_transform_gradient<float, 5, 2>(getPointer<float>(matrices), getPointer<float>(gradient), batch_size, height, width, filters,
						workspace);
		}
	}
	void cpu_kernel_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
			void *update)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		if (kernel_size == 3)
		{
			if (tile_size == 2)
				kernel_transform_update<float, 3, 2>(getPointer<float>(matrices), getPointer<float>(update), filters_out, filters_in);
			if (tile_size == 4)
				kernel_transform_update<float, 3, 4>(getPointer<float>(matrices), getPointer<float>(update), filters_out, filters_in);
		}
		if (kernel_size == 5)
		{
			if (tile_size == 2)
				kernel_transform_update<float, 5, 2>(getPointer<float>(matrices), getPointer<float>(update), filters_out, filters_in);
		}
	}

} /* namespace Vector_NAMESPACE */
