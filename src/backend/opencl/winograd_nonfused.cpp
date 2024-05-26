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

	cl::Kernel get_kernel(ml::mlContext_t context, const char *name, int kernel_size, int tile_size)
	{
		if (kernel_size == 3)
		{
			if (tile_size == 2)
			{
				static const ml::opencl::ProgramCache result("winograd_transforms_3x3_2x2",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
								+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
				return result.getKernel(context, name);
			}
			if (tile_size == 4)
			{
				static const ml::opencl::ProgramCache result("winograd_transforms_3x3_4x4",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
								+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
				return result.getKernel(context, name);
			}
		}
		if (kernel_size == 5 and tile_size == 2)
		{
			static const ml::opencl::ProgramCache result("winograd_transforms_5x5_2x2",
					ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles
							+ get_defines(kernel_size, tile_size) + ml::opencl::kernels::winograd_nonfused, "");
			return result.getKernel(context, name);
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

		cl::Kernel kernel = get_kernel(context, "transform_weights", kernel_size, tile_size);
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads, filters_out);

		kernel.setArg(0, opencl::getMemoryObject(matrices).buffer());
		kernel.setArg(1, opencl::getMemoryObject(weights).buffer());
		kernel.setArg(2, filters_out);
		kernel.setArg(3, filters_in);
		kernel.setArg(4, static_cast<int>(invert));

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filters = input_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		const int max_threads = opencl::has_fp16_math(context) ? 64 : 128;

		cl::Kernel kernel = get_kernel(context, "transform_input", kernel_size, tile_size);
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads * tiles_h, tiles_w, input_shape.dim[0]);

		kernel.setArg(0, opencl::getMemoryObject(matrices).buffer());
		kernel.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel.setArg(2, batch_size);
		kernel.setArg(3, height);
		kernel.setArg(4, width);
		kernel.setArg(5, filters);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const int batch_size = output_shape.dim[0];
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int filters = output_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		const int max_threads = opencl::has_fp16_math(context) ? 64 : 128;

		cl::Kernel kernel = get_kernel(context, "transform_output", kernel_size, tile_size);
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads * tiles_h, tiles_w, batch_size);

		cl::Buffer &workspace = opencl::Context::getWorkspace(context);

		const bool use_add = (add != nullptr);
		const bool use_bias = (bias != nullptr);

		kernel.setArg(0, opencl::getMemoryObject(matrices).buffer());
		kernel.setArg(1, opencl::getMemoryObject(output).buffer());
		kernel.setArg(2, use_add ? opencl::getMemoryObject(add).buffer() : workspace);
		kernel.setArg(3, static_cast<int>(use_add));
		kernel.setArg(4, use_bias ? opencl::getMemoryObject(bias).buffer() : workspace);
		kernel.setArg(5, static_cast<int>(use_bias));
		kernel.setArg(6, static_cast<int>(act));
		kernel.setArg(7, batch_size);
		kernel.setArg(8, height);
		kernel.setArg(9, width);
		kernel.setArg(10, filters);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
			const void *gradient, void *matrices)
	{
		const int batch_size = gradient_shape.dim[0];
		const int height = gradient_shape.dim[1];
		const int width = gradient_shape.dim[2];
		const int filters = gradient_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);

		const int tiles_h = get_number_of_tiles(height, tile_size);
		const int tiles_w = get_number_of_tiles(width, tile_size);
		const int max_threads = opencl::has_fp16_math(context) ? 64 : 128;

		cl::Kernel kernel = get_kernel(context, "transform_gradientweights", kernel_size, tile_size);
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads * tiles_h, tiles_w, gradient_shape.dim[0]);

		kernel.setArg(0, opencl::getMemoryObject(matrices).buffer());
		kernel.setArg(1, opencl::getMemoryObject(gradient).buffer());
		kernel.setArg(2, batch_size);
		kernel.setArg(3, height);
		kernel.setArg(4, width);
		kernel.setArg(5, filters);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
			void *update)
	{
		const int filters_out = weight_shape.dim[0];
		const int filters_in = weight_shape.dim[3];

		const int kernel_size = get_kernel_size(weight_shape);
		const int max_threads = opencl::has_fp16_math(context) ? 64 : 128;

		cl::Kernel kernel = get_kernel(context, "transform_update", kernel_size, tile_size);
		cl::NDRange local(max_threads);
		cl::NDRange global(max_threads, filters_out);

		kernel.setArg(0, opencl::getMemoryObject(matrices).buffer());
		kernel.setArg(1, opencl::getMemoryObject(update).buffer());
		kernel.setArg(2, filters_out);
		kernel.setArg(3, filters_in);

		opencl::runKernel(context, kernel, global, local);
	}

} /* namespace ml */

