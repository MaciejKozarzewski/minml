/*
 * depthwise_conv.cpp
 *
 *  Created on: Jun 9, 2025
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

	std::string get_defines(int kernel_size)
	{
		return "#define KERNEL_SIZE " + std::to_string(kernel_size) + "\n";
	}

	cl::Kernel get_kernel(ml::mlContext_t context, const char *name, int kernel_size)
	{
		switch (kernel_size)
		{
			case 3:
			{
				static const ml::opencl::ProgramCache result("depthwise_conv_forward_3x3",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles + get_defines(kernel_size)
								+ ml::opencl::kernels::depthwise_conv, "");
				return result.getKernel(context, name);
			}
			case 5:
			{
				static const ml::opencl::ProgramCache result("depthwise_conv_forward_5x5",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles + get_defines(kernel_size)
								+ ml::opencl::kernels::depthwise_conv, "");
				return result.getKernel(context, name);
			}
			case 7:
			{
				static const ml::opencl::ProgramCache result("depthwise_conv_forward_7x7",
						ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::lines_and_tiles + get_defines(kernel_size)
								+ ml::opencl::kernels::depthwise_conv, "");
				return result.getKernel(context, name);
			}
		}
		throw std::logic_error("Unsupported configuration of kernel size = " + std::to_string(kernel_size));
	}

}

namespace ml
{
	void opencl_depthwise_conv_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(w.rank == 3);
		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int filter_size = w.dim[0];
		const int channels = w.dim[2];
		assert(w.dim[0] == w.dim[1]);

		constexpr int TileSize = 4;

		const int num_tiles_h = (height + TileSize - 1) / TileSize;

		cl::Kernel kernel = get_kernel(context, "depthwise_conv_forward", filter_size);
		cl::NDRange local(32, TileSize);
		cl::NDRange global(32 * ((channels + 32 - 1) / 32), TileSize * num_tiles_h, batch_size);

		kernel.setArg(0, beta);
		kernel.setArg(1, opencl::getMemoryObject(y.data).buffer());
		kernel.setArg(2, alpha);
		kernel.setArg(3, opencl::getMemoryObject(x.data).buffer());
		kernel.setArg(4, opencl::getMemoryObject(w.data).buffer());
		kernel.setArg(5, opencl::getMemoryObject(b.data).buffer());
		kernel.setArg(6, height);
		kernel.setArg(7, width);
		kernel.setArg(8, channels);
		kernel.setArg(9, static_cast<int>(false));

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_depthwise_conv_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t w, float beta, mlTensor_t dx)
	{
	}
	void opencl_depthwise_conv_update(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t dy, float beta, mlTensor_t dw)
	{
	}

} /* namespace ml */

