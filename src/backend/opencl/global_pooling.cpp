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
	const cl::Program& get_program()
	{
		static const cl::Program program = ml::opencl::compileProgram("global pooling",
				ml::opencl::kernels::common + ml::opencl::kernels::indexers + ml::opencl::kernels::global_pooling, "");
		return program;

	}
}

namespace ml
{

	void opencl_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		cl::Kernel kernel = opencl::getKernel(get_program(), "pooling_avg_max_forward");
		cl::NDRange global(32 * ((dim2 + 31) / 32), 32, dim0);
		cl::NDRange local(32, 32);

		kernel.setArg(0, opencl::getBuffer(output));
		kernel.setArg(1, opencl::getBuffer(input));
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

		cl::Kernel kernel = opencl::getKernel(get_program(), "pooling_avg_max_backward");
		cl::NDRange global(128 * ((dim2 + 127) / 128), std::max(256, dim1), dim0);
		cl::NDRange local(128);

		kernel.setArg(0, opencl::getBuffer(gradient_prev));
		kernel.setArg(1, opencl::getBuffer(gradient_next));
		kernel.setArg(2, opencl::getBuffer(input));
		kernel.setArg(3, opencl::getBuffer(output));
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

		cl::Kernel kernel = opencl::getKernel(get_program(), "global_broadcast_forward");
		cl::NDRange global(128 * ((dim2 + 127) / 128), std::max(256, dim1), dim0);
		cl::NDRange local(128);

		kernel.setArg(0, opencl::getBuffer(output));
		kernel.setArg(1, opencl::getBuffer(input));
		kernel.setArg(2, opencl::getBuffer(bias));
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

		cl::Kernel kernel = opencl::getKernel(get_program(), "global_broadcast_backward");
		cl::NDRange global(32 * ((dim2 + 31) / 32), 32, dim0);
		cl::NDRange local(32, 32);

		kernel.setArg(0, opencl::getBuffer(gradient_prev));
		kernel.setArg(1, opencl::getBuffer(gradient_next));
		kernel.setArg(2, opencl::getBuffer(output));
		kernel.setArg(3, dim0);
		kernel.setArg(4, dim1);
		kernel.setArg(5, dim2);
		kernel.setArg(6, static_cast<int>(act));

		opencl::runKernel(context, kernel, global, local);
	}
} /* namespace ml */

