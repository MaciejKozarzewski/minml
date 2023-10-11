/*
 * winograd_runtime.cpp
 *
 *  Created on: Sep 30, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.hpp"
#include "winograd_kernels.hpp"
#include "winograd_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstring>
#include <omp.h>

namespace ml
{
	template<size_t Alignment>
	size_t round_up_to_multiple_of(size_t x) noexcept
	{
		const size_t remainder = x % Alignment;
		const size_t shift = (remainder == 0) ? 0 : (Alignment - remainder);
		return x + shift;
	}
	template<size_t Alignment, typename T>
	T* align_pointer_to(T *ptr) noexcept
	{
		const size_t remainder = reinterpret_cast<std::uintptr_t>(ptr) % Alignment;
		const size_t shift = (remainder == 0) ? 0 : (Alignment - remainder);
		return reinterpret_cast<uint8_t*>(ptr) + shift;
	}

	WinogradTransformRuntime::WinogradTransformRuntime(int tileSize, int kernelSize, mlDataType_t dtype) noexcept :
			tile_size(tileSize),
			kernel_size(kernelSize),
			dtype(dtype)
	{
		setup_default_transforms();
	}
	void WinogradTransformRuntime::transformWeights(mlContext_t context, mlShape_t shape, const void *weights, void *matrices, bool invert) const
	{
		assert(weight_transform != nullptr);
		void *workspace = cpu::Context::getWorkspace(context);
		const int output_filters = shape.dim[0];
		const int kernel_height = shape.dim[1];
		const int kernel_width = shape.dim[2];
		const int input_filters = shape.dim[3];

		const int TransformSize = tile_size + kernel_size - 1;

		const Indexer<3> matrices_indexer(TransformSize * TransformSize, output_filters, input_filters * size_of(dtype));
		const Indexer<4> input_indexer(output_filters, kernel_height, kernel_width, input_filters * size_of(dtype));

		const size_t ptr_in_size = round_up_to_multiple_of<64>(sizeof(void*) * kernel_height * kernel_width);
		const size_t ptr_out_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t workspace_size = round_up_to_multiple_of<64>(SimdSize * TransformSize * TransformSize + WorkspaceAlignment);

#pragma omp parallel
		{
			uint8_t *local_workspace = reinterpret_cast<uint8_t*>(workspace) + omp_get_thread_num() * (ptr_in_size + ptr_out_size + workspace_size);
			const void **ptr_in = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_in_size;

			void **ptr_out = reinterpret_cast<void**>(local_workspace);
			local_workspace += ptr_out_size;

			void *transform_workspace = align_pointer_to<WorkspaceAlignment>(local_workspace);

#pragma omp for
			for (int tile_idx = 0; tile_idx < output_filters; tile_idx++)
			{
				for (int i = 0; i < TransformSize * TransformSize; i++)
					ptr_out[i] = getPointer<uint8_t>(matrices) + matrices_indexer.at(i, tile_idx, 0);

				int matrix_idx = 0;
				for (int i = 0; i < kernel_height; i++)
					for (int j = 0; j < kernel_width; j++, matrix_idx++)
					{
						const int x = invert ? (kernel_height - 1 - i) : i;
						const int y = invert ? (kernel_width - 1 - j) : j;
						ptr_in[matrix_idx] = getPointer<uint8_t>(weights) + input_indexer.at(tile_idx, x, y, 0);
					}
				weight_transform(ptr_in, ptr_out, transform_workspace, input_filters);
			}
		}
	}
	void WinogradTransformRuntime::transformInput(mlContext_t context, mlShape_t shape, const void *input, void *matrices) const
	{
		assert(input_transform != nullptr);
		void *workspace = cpu::Context::getWorkspace(context);
		const int batch_size = shape.dim[0];
		const int height = shape.dim[1];
		const int width = shape.dim[2];
		const int filters = shape.dim[3];

		const int Padding = kernel_size / 2;
		const int TransformSize = tile_size + kernel_size - 1;

		const int tiles_h = (height + tile_size - 1) / tile_size;
		const int tiles_w = (width + tile_size - 1) / tile_size;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, filters * size_of(dtype));
		const Indexer<4> input_indexer(batch_size, height, width, filters * size_of(dtype));

		const size_t zero_line_size = round_up_to_multiple_of<64>(size_of(dtype) * filters);
		const size_t ptr_in_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t ptr_out_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t workspace_size = round_up_to_multiple_of<64>(SimdSize * TransformSize * TransformSize + WorkspaceAlignment);

		void *zero_line = workspace;
		std::memset(zero_line, 0, zero_line_size);

//#pragma omp parallel
		{
			uint8_t *local_workspace = reinterpret_cast<uint8_t*>(workspace) + zero_line_size
					+ omp_get_thread_num() * (ptr_in_size + ptr_out_size + workspace_size);
			const void **ptr_in = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_in_size;

			void **ptr_out = reinterpret_cast<void**>(local_workspace);
			local_workspace += ptr_out_size;

			void *transform_workspace = align_pointer_to<WorkspaceAlignment>(local_workspace);

//#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				const int batch = tile_idx / tiles_per_image;
				const int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				const int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				int matrix_idx = 0;
				for (int i = 0; i < TransformSize; i++)
					for (int j = 0; j < TransformSize; j++, matrix_idx++)
					{
						const int x = tile_size * tile_x + i - Padding;
						const int y = tile_size * tile_y + j - Padding;
						ptr_out[matrix_idx] = getPointer<uint8_t>(matrices) + matrices_indexer.at(matrix_idx, tile_idx, 0);
						if (x >= 0 and x < height and y >= 0 and y < width)
							ptr_in[matrix_idx] = getPointer<uint8_t>(input) + input_indexer.at(batch, x, y, 0);
						else
							ptr_in[matrix_idx] = zero_line;
					}
				input_transform(ptr_in, ptr_out, transform_workspace, filters);
			}
		}
	}
	void WinogradTransformRuntime::transformOutput(mlContext_t context, mlShape_t shape, const void *matrices, void *output, const void *bias,
			const void *ext, bool use_relu) const
	{
		assert(output_transform != nullptr);
		void *workspace = cpu::Context::getWorkspace(context);
		const int batch_size = shape.dim[0];
		const int height = shape.dim[1];
		const int width = shape.dim[2];
		const int filters = shape.dim[3];

		const int TransformSize = tile_size + kernel_size - 1;

		const int tiles_h = (height + tile_size - 1) / tile_size;
		const int tiles_w = (width + tile_size - 1) / tile_size;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, filters * size_of(dtype));
		const Indexer<4> output_indexer(batch_size, height, width, filters * size_of(dtype));

		const size_t fake_storage_size = round_up_to_multiple_of<64>(size_of(dtype) * filters);
		const size_t ptr_in_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t ptr_out_size = round_up_to_multiple_of<64>(sizeof(void*) * tile_size * tile_size);
		const size_t workspace_size = round_up_to_multiple_of<64>(SimdSize * TransformSize * TransformSize + WorkspaceAlignment);

//#pragma omp parallel
		{
			uint8_t *local_workspace = reinterpret_cast<uint8_t*>(workspace)
					+ omp_get_thread_num() * (fake_storage_size + ptr_in_size + 2 * ptr_out_size + workspace_size);
			void *fake_storage = reinterpret_cast<void*>(local_workspace);
			local_workspace += fake_storage_size;

			const void **ptr_in = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_in_size;

			void **ptr_out = reinterpret_cast<void**>(local_workspace);
			local_workspace += ptr_out_size;

			const void **ptr_ext = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_out_size;

			void *transform_workspace = align_pointer_to<WorkspaceAlignment>(local_workspace);

//#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				const int batch = tile_idx / tiles_per_image;
				const int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				const int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				for (int i = 0; i < TransformSize * TransformSize; i++)
					ptr_in[i] = getPointer<uint8_t>(matrices) + matrices_indexer.at(i, tile_idx, 0);

				int matrix_idx = 0;
				for (int i = 0; i < tile_size; i++)
					for (int j = 0; j < tile_size; j++, matrix_idx++)
					{
						const int x = tile_size * tile_x + i;
						const int y = tile_size * tile_y + j;
						if (x < height and y < width)
						{
							ptr_out[matrix_idx] = getPointer<uint8_t>(output) + output_indexer.at(batch, x, y, 0);
							if (ext != nullptr)
								ptr_ext[matrix_idx] = getPointer<uint8_t>(ext) + output_indexer.at(batch, x, y, 0);
						}
						else
						{
							ptr_out[matrix_idx] = fake_storage;
							ptr_ext[matrix_idx] = fake_storage;
						}
					}
				if (ext != nullptr)
					output_transform(ptr_in, ptr_out, transform_workspace, filters, ptr_ext, bias, use_relu);
				else
					output_transform(ptr_in, ptr_out, transform_workspace, filters, nullptr, bias, use_relu);
			}
		}
	}
	void WinogradTransformRuntime::transformGradient(mlContext_t context, mlShape_t shape, const void *gradient, void *matrices) const
	{
		assert(gradient_transform != nullptr);
		void *workspace = cpu::Context::getWorkspace(context);
		const int batch_size = shape.dim[0];
		const int height = shape.dim[1];
		const int width = shape.dim[2];
		const int filters = shape.dim[3];

		const int TransformSize = tile_size + kernel_size - 1;

		const int tiles_h = (height + tile_size - 1) / tile_size;
		const int tiles_w = (width + tile_size - 1) / tile_size;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;
		const Indexer<3> matrices_indexer(TransformSize * TransformSize, nb_of_tiles, filters * size_of(dtype));
		const Indexer<4> input_indexer(batch_size, height, width, filters * size_of(dtype));

		const size_t zero_line_size = round_up_to_multiple_of<64>(size_of(dtype) * filters);
		const size_t ptr_in_size = round_up_to_multiple_of<64>(sizeof(void*) * tile_size * tile_size);
		const size_t ptr_out_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t workspace_size = round_up_to_multiple_of<64>(SimdSize * TransformSize * TransformSize + WorkspaceAlignment);

		void *zero_line = workspace;
		std::memset(zero_line, 0, zero_line_size);

#pragma omp parallel
		{
			uint8_t *local_workspace = reinterpret_cast<uint8_t*>(workspace) + zero_line_size
					+ omp_get_thread_num() * (ptr_in_size + ptr_out_size + workspace_size);
			const void **ptr_in = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_in_size;

			void **ptr_out = reinterpret_cast<void**>(local_workspace);
			local_workspace += ptr_out_size;

			void *transform_workspace = align_pointer_to<WorkspaceAlignment>(local_workspace);

#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				const int batch = tile_idx / tiles_per_image;
				const int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				const int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				for (int i = 0; i < TransformSize * TransformSize; i++)
					ptr_out[i] = getPointer<uint8_t>(matrices) + matrices_indexer.at(i, tile_idx, 0);

				int matrix_idx = 0;
				for (int i = 0; i < tile_size; i++)
					for (int j = 0; j < tile_size; j++, matrix_idx++)
					{
						const int x = tile_size * tile_x + i;
						const int y = tile_size * tile_y + j;
						if (x < height and y < width)
							ptr_in[matrix_idx] = getPointer<uint8_t>(gradient) + input_indexer.at(batch, x, y, 0);
						else
							ptr_in[matrix_idx] = zero_line;
					}
				gradient_transform(ptr_in, ptr_out, transform_workspace, filters);
			}
		}
	}
	void WinogradTransformRuntime::transformUpdate(mlContext_t context, mlShape_t shape, const void *matrices, void *update) const
	{
		assert(update_transform != nullptr);
		void *workspace = cpu::Context::getWorkspace(context);
		const int output_filters = shape.dim[0];
		const int kernel_height = shape.dim[1];
		const int kernel_width = shape.dim[2];
		const int input_filters = shape.dim[3];

		const int TransformSize = tile_size + kernel_size - 1;

		const Indexer<3> matrices_indexer(TransformSize * TransformSize, output_filters, input_filters * size_of(dtype));
		const Indexer<4> input_indexer(output_filters, kernel_height, kernel_width, input_filters * size_of(dtype));

		const size_t ptr_in_size = round_up_to_multiple_of<64>(sizeof(void*) * TransformSize * TransformSize);
		const size_t ptr_out_size = round_up_to_multiple_of<64>(sizeof(void*) * kernel_height * kernel_width);
		const size_t workspace_size = round_up_to_multiple_of<64>(SimdSize * TransformSize * TransformSize + WorkspaceAlignment);

#pragma omp parallel
		{
			uint8_t *local_workspace = reinterpret_cast<uint8_t*>(workspace) + omp_get_thread_num() * (ptr_in_size + ptr_out_size + workspace_size);
			const void **ptr_in = reinterpret_cast<const void**>(local_workspace);
			local_workspace += ptr_in_size;

			void **ptr_out = reinterpret_cast<void**>(local_workspace);
			local_workspace += ptr_out_size;

			void *transform_workspace = align_pointer_to<WorkspaceAlignment>(local_workspace);

#pragma omp for
			for (int tile_idx = 0; tile_idx < output_filters; tile_idx++)
			{
				for (int i = 0; i < TransformSize * TransformSize; i++)
					ptr_in[i] = getPointer<uint8_t>(matrices) + matrices_indexer.at(i, tile_idx, 0);

				int matrix_idx = 0;
				for (int i = 0; i < kernel_height; i++)
					for (int j = 0; j < kernel_width; j++, matrix_idx++)
						ptr_out[matrix_idx] = getPointer<uint8_t>(update) + input_indexer.at(tile_idx, i, j, 0);
				update_transform(ptr_in, ptr_out, transform_workspace, input_filters);
			}
		}
	}
	/*
	 * private
	 */
	void WinogradTransformRuntime::setup_default_transforms() noexcept
	{
		if (kernel_size == 3 and tile_size == 4)
		{
			if (dtype == DTYPE_FLOAT16)
			{
				weight_transform = winograd_weight_transform_4x4_3x3_def_fp16;
				input_transform = winograd_input_transform_4x4_3x3_def_fp16;
				output_transform = winograd_output_transform_4x4_3x3_def_fp16;
			}
			if (dtype == DTYPE_FLOAT32)
			{
				weight_transform = winograd_weight_transform_4x4_3x3_def_fp32;
				input_transform = winograd_input_transform_4x4_3x3_def_fp32;
				output_transform = winograd_output_transform_4x4_3x3_def_fp32;
			}
			gradient_transform = winograd_gradient_transform_4x4_3x3_def_fp32;
			update_transform = winograd_update_transform_4x4_3x3_def_fp32;
		}
		if (kernel_size == 3 and tile_size == 5)
		{
			if (dtype == DTYPE_FLOAT16)
			{
				weight_transform = winograd_weight_transform_5x5_3x3_def_fp16;
				input_transform = winograd_input_transform_5x5_3x3_def_fp16;
				output_transform = winograd_output_transform_5x5_3x3_def_fp16;
			}
			if (dtype == DTYPE_FLOAT32)
			{
				weight_transform = winograd_weight_transform_5x5_3x3_def_fp32;
				input_transform = winograd_input_transform_5x5_3x3_def_fp32;
				output_transform = winograd_output_transform_5x5_3x3_def_fp32;
			}
		}
		if (kernel_size == 5 and tile_size == 2)
		{
			if (dtype == DTYPE_FLOAT32)
			{
				weight_transform = winograd_weight_transform_2x2_5x5_def_fp32;
				input_transform = winograd_input_transform_2x2_5x5_def_fp32;
				output_transform = winograd_output_transform_2x2_5x5_def_fp32;
			}
			gradient_transform = winograd_gradient_transform_2x2_5x5_def_fp32;
			update_transform = winograd_update_transform_2x2_5x5_def_fp32;
		}
	}

	std::vector<WinogradTransformRuntime> get_sse2_winograd_runtime()
	{
		std::vector<WinogradTransformRuntime> result(2);
		// 4x4 tile, 3x3 kernel, fp32
		result[0] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT32);
		result[0].input_transform = winograd_input_transform_4x4_3x3_sse2_fp32;
		result[0].output_transform = winograd_output_transform_4x4_3x3_sse2_fp32;

		// 4x4 tile, 3x3 kernel, fp32
		result[1] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT32);
		result[1].input_transform = winograd_input_transform_5x5_3x3_sse2_fp32;
		result[1].output_transform = winograd_output_transform_5x5_3x3_sse2_fp32;

		return result;
	}
	std::vector<WinogradTransformRuntime> get_avx_winograd_runtime()
	{
		std::vector<WinogradTransformRuntime> result(4);
		// 4x4 tile, 3x3 kernel, fp32
		result[0] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT32);
		result[0].input_transform = winograd_input_transform_4x4_3x3_avx_fp32;
		result[0].output_transform = winograd_output_transform_4x4_3x3_avx_fp32;

		// 4x4 tile, 3x3 kernel, fp16
		result[1] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT16);
		result[1].input_transform = winograd_input_transform_4x4_3x3_avx_fp16;
		result[1].output_transform = winograd_output_transform_4x4_3x3_avx_fp16;

		// 5x5 tile, 3x3 kernel, fp32
		result[2] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT32);
		result[2].input_transform = winograd_input_transform_5x5_3x3_avx_fp32;
		result[2].output_transform = winograd_output_transform_5x5_3x3_avx_fp32;

		// 5x5 tile, 3x3 kernel, fp16
		result[3] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT16);
		result[3].input_transform = winograd_input_transform_5x5_3x3_avx_fp16;
		result[3].output_transform = winograd_output_transform_5x5_3x3_avx_fp16;

		return result;
	}
	std::vector<WinogradTransformRuntime> get_avx2_fma_winograd_runtime()
	{
		std::vector<WinogradTransformRuntime> result(4);
		// 4x4 tile, 3x3 kernel, fp32
		result[0] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT32);
		result[0].input_transform = winograd_input_transform_4x4_3x3_avx2_fma_fp32;
		result[0].output_transform = winograd_output_transform_4x4_3x3_avx2_fma_fp32;

		// 4x4 tile, 3x3 kernel, fp16
		result[1] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT16);
		result[1].input_transform = winograd_input_transform_4x4_3x3_avx2_fma_fp16;
		result[1].output_transform = winograd_output_transform_4x4_3x3_avx2_fma_fp16;

		// 5x5 tile, 3x3 kernel, fp32
		result[2] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT32);
		result[2].input_transform = winograd_input_transform_5x5_3x3_avx2_fma_fp32;
		result[2].output_transform = winograd_output_transform_5x5_3x3_avx2_fma_fp32;

		// 5x5 tile, 3x3 kernel, fp16
		result[3] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT16);
		result[3].input_transform = winograd_input_transform_5x5_3x3_avx2_fma_fp16;
		result[3].output_transform = winograd_output_transform_5x5_3x3_avx2_fma_fp16;

		return result;
	}
	std::vector<WinogradTransformRuntime> get_avx512f_winograd_runtime()
	{
		std::vector<WinogradTransformRuntime> result(4);
		// 4x4 tile, 3x3 kernel, fp32
		result[0] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT32);
		result[0].input_transform = winograd_input_transform_4x4_3x3_avx512f_fp32;
		result[0].output_transform = winograd_output_transform_4x4_3x3_avx512f_fp32;

		// 4x4 tile, 3x3 kernel, fp16
		result[1] = WinogradTransformRuntime(4, 3, DTYPE_FLOAT16);
		result[1].input_transform = winograd_input_transform_4x4_3x3_avx512f_fp16;
		result[1].output_transform = winograd_output_transform_4x4_3x3_avx512f_fp16;

		// 5x5 tile, 3x3 kernel, fp32
		result[2] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT32);
		result[2].input_transform = winograd_input_transform_5x5_3x3_avx512f_fp32;
		result[2].output_transform = winograd_output_transform_5x5_3x3_avx512f_fp32;

		// 5x5 tile, 3x3 kernel, fp16
		result[3] = WinogradTransformRuntime(5, 3, DTYPE_FLOAT16);
		result[3].input_transform = winograd_input_transform_5x5_3x3_avx512f_fp16;
		result[3].output_transform = winograd_output_transform_5x5_3x3_avx512f_fp16;

		return result;
	}

	std::vector<WinogradTransformRuntime> create_winograd_runtime_table(mlContext_t context)
	{
		const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
		if (simd >= cpu::SimdLevel::AVX512F)
			return get_avx512f_winograd_runtime();
		if (simd >= cpu::SimdLevel::AVX2)
			return get_avx2_fma_winograd_runtime();
		if (simd >= cpu::SimdLevel::AVX)
			return get_avx_winograd_runtime();
		if (simd >= cpu::SimdLevel::SSE2)
			return get_sse2_winograd_runtime();
		return std::vector<WinogradTransformRuntime>();
	}
	const WinogradTransformRuntime& get_runtime(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape)
	{
		assert(weight_shape.rank == 4);
		assert(weight_shape.dim[1] == weight_shape.dim[2]); // only for square kernels
		static const std::vector<WinogradTransformRuntime> table = create_winograd_runtime_table(context);

		const int kernel_size = weight_shape.dim[1];

		for (auto iter = table.begin(); iter < table.end(); iter++)
			if (iter->tile_size == tile_size and iter->kernel_size == kernel_size and iter->dtype == dtype)
				return *iter;
		throw std::runtime_error("No suitable Winograd transform runtime");
	}

	void cpu_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
			void *matrices, bool invert)
	{
		get_runtime(context, tile_size, dtype, weight_shape).transformWeights(context, weight_shape, weights, matrices, invert);
	}
	void cpu_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		get_runtime(context, tile_size, dtype, weight_shape).transformInput(context, input_shape, input, matrices);
	}
	void cpu_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const bool use_relu = (act == ACTIVATION_RELU);
		get_runtime(context, tile_size, dtype, weight_shape).transformOutput(context, output_shape, matrices, output, bias, add, use_relu);
		if (not use_relu)
			cpu_activation_forward(context, dtype, output_shape, output, output, act);
	}
//	void cpu_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
//			const void *gradient, void *matrices)
//	{
//		get_runtime(context, tile_size, dtype, weight_shape).transformGradient(context, gradient_shape, gradient, matrices);
//	}
//	void cpu_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
//			void *update)
//	{
//		get_runtime(context, tile_size, dtype, weight_shape).transformUpdate(context, weight_shape, matrices, update);
//	}

} /* namespace ml */

