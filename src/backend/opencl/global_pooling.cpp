/*
 * global_pooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	cl::Kernel get_kernel(ml::mlContext_t context, const char *name)
	{
		static const ml::opencl::ProgramCache result("global pooling",
				ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::global_pooling, "");
		return result.getKernel(context, name);
	}
}

namespace ml
{

	void opencl_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cl::Kernel kernel = get_kernel(context, "pooling_avg_max_forward");
		cl::NDRange global(32 * ((dim2 + 31) / 32), 32, dim0);
		cl::NDRange local(32, 32);

		kernel.setArg(0, opencl::getMemoryObject(output).buffer());
		kernel.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel.setArg(2, dim0);
		kernel.setArg(3, dim1);
		kernel.setArg(4, dim2);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input, const void *output)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cl::Kernel kernel = get_kernel(context, "pooling_avg_max_backward");
		cl::NDRange global(128 * ((dim2 + 127) / 128), std::max(256, dim1), dim0);
		cl::NDRange local(128);

		kernel.setArg(0, opencl::getMemoryObject(gradient_prev).buffer());
		kernel.setArg(1, opencl::getMemoryObject(gradient_next).buffer());
		kernel.setArg(2, opencl::getMemoryObject(input).buffer());
		kernel.setArg(3, opencl::getMemoryObject(output).buffer());
		kernel.setArg(4, dim0);
		kernel.setArg(5, dim1);
		kernel.setArg(6, dim2);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
			const void *bias, mlActivationType_t act)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cl::Kernel kernel = get_kernel(context, "global_broadcast_forward");
		cl::NDRange global(128 * ((dim2 + 127) / 128), std::max(256, dim1), dim0);
		cl::NDRange local(128);

		kernel.setArg(0, opencl::getMemoryObject(output).buffer());
		kernel.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel.setArg(2, opencl::getMemoryObject(bias).buffer());
		kernel.setArg(3, dim0);
		kernel.setArg(4, dim1);
		kernel.setArg(5, dim2);
		kernel.setArg(6, static_cast<int>(act));

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cl::Kernel kernel = get_kernel(context, "global_broadcast_backward");
		cl::NDRange global(32 * ((dim2 + 31) / 32), 32, dim0);
		cl::NDRange local(32, 32);

		kernel.setArg(0, opencl::getMemoryObject(gradient_prev).buffer());
		kernel.setArg(1, opencl::getMemoryObject(gradient_next).buffer());
		kernel.setArg(2, opencl::getMemoryObject(output).buffer());
		kernel.setArg(3, dim0);
		kernel.setArg(4, dim1);
		kernel.setArg(5, dim2);
		kernel.setArg(6, static_cast<int>(act));

		opencl::runKernel(context, kernel, global, local);
	}
} /* namespace ml */

