/*
 * winograd_nonfused.cpp
 *
 *  Created on: Nov 18, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <CL/opencl.hpp>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <iostream>

namespace
{
	int get_kernel_size(const ml::mlShape_t &weight_shape) noexcept
	{
		assert(weight_shape.rank == 4);
		assert(weight_shape.dim[1] == weight_shape.dim[2]);
		return weight_shape.dim[1];
	}
	int get_number_of_tiles(int dim, int transform_size) noexcept
	{
		return (dim + transform_size - 1) / transform_size;
	}

	std::string get_defines(int kernel_size, int tile_size)
	{
		return "#define KERNEL_SIZE " + std::to_string(kernel_size) + "\n" + "#define TRANSFORM_SIZE " + std::to_string(tile_size) + "\n";
	}

	const cl::Program& get_program(int kernel_size, int tile_size)
	{
		if (kernel_size == 3)
		{
			if (tile_size == 2)
			{
				const static cl::Program program_3x3_2x2 = ml::opencl::compile_program("winograd_transforms_3x3_2x2",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
								+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
				return program_3x3_2x2;
			}
			if (tile_size == 4)
			{
				const static cl::Program program_3x3_4x4 = ml::opencl::compile_program("winograd_transforms_3x3_4x4",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
								+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
				return program_3x3_4x4;
			}
		}
		if (kernel_size == 5 and tile_size == 2)
		{
			const static cl::Program program_5x5_2x2 = ml::opencl::compile_program("winograd_transforms_5x5_2x2",
					ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
							+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
			return program_5x5_2x2;
		}
		throw std::logic_error(
				"Unsupported configuration of kernel size = " + std::to_string(kernel_size) + " and tile size = " + std::to_string(tile_size));
	}

}

namespace ml
{
	void opencl_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
			void *matrices, bool invert)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int max_threads = opencl::has_fp16_math(context) ? 64 : 128;

		cl::Kernel kernel(get_program(kernel_size, tile_size), "transform_weights");
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads, filters_out);

		kernel.setArg(0, opencl::get_buffer(matrices));
		kernel.setArg(1, opencl::get_buffer(weights));
		kernel.setArg(2, filters_out);
		kernel.setArg(3, filters_in);
		kernel.setArg(4, static_cast<int>(invert));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
	}
	void opencl_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
	}
	void opencl_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
			const void *gradient, void *matrices)
	{
//		const int batch_size = gradient_shape.dim[0];
//		const int height = gradient_shape.dim[1];
//		const int width = gradient_shape.dim[2];
//		const int filters = gradient_shape.dim[3];
//
//		const int kernel_size = get_kernel_size(weight_shape);
//
//		const int tiles_h = get_number_of_tiles(height, tile_size);
//		const int tiles_w = get_number_of_tiles(width, tile_size);
//		cudaStream_t stream = cuda::Context::getStream(context);
//
//		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
//		dim3 blockSize(std::min(max_threads, filters));
//		dim3 gridSize(tiles_h, tiles_w, gradient_shape.dim[0]);
//
//		if (kernel_size == 3)
//		{
//			if (tile_size == 2)
//				kernel_transform_gradient<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient),
//						batch_size, height, width, filters);
//			if (tile_size == 4)
//				kernel_transform_gradient<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient),
//						batch_size, height, width, filters);
//		}
//		if (kernel_size == 5 && tile_size == 2)
//		{
//			kernel_transform_gradient<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(gradient), batch_size,
//					height, width, filters);
//		}
//		assert(cudaGetLastError() == cudaSuccess);
	}
	void opencl_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
			void *update)
	{
//		const int filters_out = weight_shape.dim[0];
//		const int filters_in = weight_shape.dim[3];
//
//		const int kernel_size = get_kernel_size(weight_shape);
//
//		const int max_threads = cuda::has_fp16_math(context) ? 64 : 128;
//		dim3 blockSize(std::min(max_threads, filters_in));
//		dim3 gridSize(1, filters_out);
//		cudaStream_t stream = cuda::Context::getStream(context);
//
//		if (kernel_size == 3)
//		{
//			if (tile_size == 2)
//				kernel_transform_update<3, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update),
//						filters_out, filters_in);
//			if (tile_size == 4)
//				kernel_transform_update<3, 4> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update),
//						filters_out, filters_in);
//		}
//		if (kernel_size == 5 && tile_size == 2)
//		{
//			kernel_transform_update<5, 2> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matrices), getPointer<float>(update), filters_out,
//					filters_in);
//		}
//
//		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

