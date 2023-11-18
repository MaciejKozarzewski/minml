/*
 * batchnorm.cpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <CL/opencl.hpp>

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	const cl::Program& get_program()
	{
		static const cl::Program program = ml::opencl::compile_program("batchnorm",
				ml::opencl::kernels::common + ml::opencl::kernels::reductions + ml::opencl::kernels::batchnorm, "");
		return program;

	}
}

namespace ml
{
	void opencl_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights,
			mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Kernel kernel(get_program(), "batchnorm_inference");
		cl::NDRange global(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local(32, 8);

		kernel.setArg(0, opencl::get_buffer(weights));
		kernel.setArg(1, opencl::get_buffer(input));
		kernel.setArg(2, opencl::get_buffer(output));
		kernel.setArg(3, first_dim);
		kernel.setArg(4, last_dim);
		kernel.setArg(5, static_cast<int>(act));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights, void *running_stats,
			int running_stat_idx, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		const int workspace_first_dim = std::min((size_t) 256, opencl::Context::getWorkspaceSize(context) / (3 * sizeof(float) * last_dim));
		assert(workspace_first_dim > 0);

		cl::NDRange local12(32, 32);

		cl::Kernel kernel1(get_program(), "batchnorm_forward_avg_var_1");
		cl::NDRange global1(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

		kernel1.setArg(0, workspace);
		kernel1.setArg(1, opencl::get_buffer(input));
		kernel1.setArg(2, first_dim);
		kernel1.setArg(3, last_dim);

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel1, cl::NullRange, global1, local12);
		assert(status == CL_SUCCESS);

		cl::Kernel kernel2(get_program(), "batchnorm_forward_avg_var_2");
		cl::NDRange global2(32 * ((last_dim + 31) / 32), 32);

		kernel2.setArg(0, opencl::get_buffer(running_stats));
		kernel2.setArg(1, running_stat_idx);
		kernel2.setArg(2, workspace);
		kernel2.setArg(3, workspace_first_dim);
		kernel2.setArg(4, last_dim);

		status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel2, cl::NullRange, global2, local12);
		assert(status == CL_SUCCESS);

		cl::Kernel kernel3(get_program(), "batchnorm_forward");
		cl::NDRange global3(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local3(32, 8);

		kernel3.setArg(0, opencl::get_buffer(weights));
		kernel3.setArg(1, opencl::get_buffer(input));
		kernel3.setArg(2, opencl::get_buffer(output));
		kernel3.setArg(3, opencl::get_buffer(running_stats));
		kernel3.setArg(4, running_stat_idx);
		kernel3.setArg(5, first_dim);
		kernel3.setArg(6, last_dim);
		kernel3.setArg(7, static_cast<int>(act));

		status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel3, cl::NullRange, global3, local3);
		assert(status == CL_SUCCESS);
	}
	void opencl_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
			void *gradient_next, const void *weights, void *weights_update, const void *running_stats, int running_stat_idx, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		const int workspace_first_dim = std::min((size_t) 256, opencl::Context::getWorkspaceSize(context) / (3 * sizeof(float) * last_dim));
		assert(workspace_first_dim > 0);

		cl::NDRange local12(32, 32);

		cl::Kernel kernel1(get_program(), "batchnorm_backward_delta_1");
		cl::NDRange global1(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

		kernel1.setArg(0, workspace);
		kernel1.setArg(1, opencl::get_buffer(input));
		kernel1.setArg(2, opencl::get_buffer(output));
		kernel1.setArg(3, opencl::get_buffer(gradient_next));
		kernel1.setArg(4, opencl::get_buffer(running_stats));
		kernel1.setArg(5, running_stat_idx);
		kernel1.setArg(6, first_dim);
		kernel1.setArg(7, last_dim);
		kernel1.setArg(8, static_cast<int>(act));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel1, cl::NullRange, global1, local12);
		assert(status == CL_SUCCESS);

		cl::Kernel kernel2(get_program(), "batchnorm_backward_delta_2");
		cl::NDRange global2(32 * ((last_dim + 31) / 32), 32);

		kernel2.setArg(0, workspace);
		kernel2.setArg(1, workspace_first_dim);
		kernel2.setArg(2, last_dim);

		status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel2, cl::NullRange, global2, local12);
		assert(status == CL_SUCCESS);

		cl::Kernel kernel3(get_program(), "batchnorm_backward");
		cl::NDRange global3(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local3(32, 8);

		kernel3.setArg(0, workspace);
		kernel3.setArg(1, opencl::get_buffer(input));
		kernel3.setArg(2, opencl::get_buffer(gradient_prev));
		kernel3.setArg(3, opencl::get_buffer(gradient_next));
		kernel3.setArg(4, opencl::get_buffer(weights));
		kernel3.setArg(5, opencl::get_buffer(weights_update));
		kernel3.setArg(6, opencl::get_buffer(running_stats));
		kernel3.setArg(7, running_stat_idx);
		kernel3.setArg(8, first_dim);
		kernel3.setArg(9, last_dim);

		status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel3, cl::NullRange, global3, local3);
		assert(status == CL_SUCCESS);
	}
	void opencl_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape) / 3;

		cl::Kernel kernel(get_program(), "batchnorm_update");
		cl::NDRange global(256 * std::max(1, (last_dim + 255) / 256));
		cl::NDRange local(256);

		kernel.setArg(0, opencl::get_buffer(running_stat));
		kernel.setArg(1, opencl::get_buffer(weights));
		kernel.setArg(2, first_dim);
		kernel.setArg(3, last_dim);
		kernel.setArg(4, static_cast<int>(use_gamma));
		kernel.setArg(5, static_cast<int>(use_beta));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = volume_without_first_dim(shape);

		cl::Kernel kernel(get_program(), "fold_batchnorm");
		cl::NDRange global(first_dim * 256);
		cl::NDRange local(256);

		kernel.setArg(0, first_dim);
		kernel.setArg(1, last_dim);
		kernel.setArg(2, opencl::get_buffer(layer_weights));
		kernel.setArg(3, opencl::get_buffer(layer_bias));
		kernel.setArg(4, opencl::get_buffer(batchnorm_weights));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}

} /* namespace ml */

