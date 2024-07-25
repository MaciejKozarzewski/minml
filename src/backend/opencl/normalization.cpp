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
	using namespace ml;

	cl::Kernel get_kernel(ml::mlContext_t context, const char *name)
	{
		static const ml::opencl::ProgramCache result("batchnorm", opencl::kernels::common + opencl::kernels::reductions + opencl::kernels::batchnorm,
				"");
		return result.getKernel(context, name);
	}
}

namespace ml
{
	void opencl_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights,
			mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Kernel kernel = get_kernel(context, "batchnorm_inference");
		cl::NDRange global(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local(32, 8);

		kernel.setArg(0, opencl::getMemoryObject(weights).buffer());
		kernel.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel.setArg(2, opencl::getMemoryObject(output).buffer());
		kernel.setArg(3, first_dim);
		kernel.setArg(4, last_dim);
		kernel.setArg(5, static_cast<int>(act));

		opencl::runKernel(context, kernel, global, local);
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

		cl::Kernel kernel1 = get_kernel(context, "batchnorm_forward_avg_var_1");
		cl::NDRange global1(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

		kernel1.setArg(0, workspace);
		kernel1.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel1.setArg(2, first_dim);
		kernel1.setArg(3, last_dim);

		opencl::runKernel(context, kernel1, global1, local12);

		cl::Kernel kernel2 = get_kernel(context, "batchnorm_forward_avg_var_2");
		cl::NDRange global2(32 * ((last_dim + 31) / 32), 32);

		kernel2.setArg(0, opencl::getMemoryObject(running_stats).buffer());
		kernel2.setArg(1, running_stat_idx);
		kernel2.setArg(2, workspace);
		kernel2.setArg(3, workspace_first_dim);
		kernel2.setArg(4, last_dim);

		opencl::runKernel(context, kernel2, global2, local12);

		cl::Kernel kernel3 = get_kernel(context, "batchnorm_forward");
		cl::NDRange global3(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local3(32, 8);

		kernel3.setArg(0, opencl::getMemoryObject(weights).buffer());
		kernel3.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel3.setArg(2, opencl::getMemoryObject(output).buffer());
		kernel3.setArg(3, opencl::getMemoryObject(running_stats).buffer());
		kernel3.setArg(4, running_stat_idx);
		kernel3.setArg(5, first_dim);
		kernel3.setArg(6, last_dim);
		kernel3.setArg(7, static_cast<int>(act));

		opencl::runKernel(context, kernel3, global3, local3);
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

		cl::Kernel kernel1 = get_kernel(context, "batchnorm_backward_delta_1");
		cl::NDRange global1(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

		kernel1.setArg(0, workspace);
		kernel1.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel1.setArg(2, opencl::getMemoryObject(output).buffer());
		kernel1.setArg(3, opencl::getMemoryObject(gradient_next).buffer());
		kernel1.setArg(4, opencl::getMemoryObject(running_stats).buffer());
		kernel1.setArg(5, running_stat_idx);
		kernel1.setArg(6, first_dim);
		kernel1.setArg(7, last_dim);
		kernel1.setArg(8, static_cast<int>(act));

		opencl::runKernel(context, kernel1, global1, local12);

		cl::Kernel kernel2 = get_kernel(context, "batchnorm_backward_delta_2");
		cl::NDRange global2(32 * ((last_dim + 31) / 32), 32);

		kernel2.setArg(0, workspace);
		kernel2.setArg(1, workspace_first_dim);
		kernel2.setArg(2, last_dim);

		opencl::runKernel(context, kernel2, global2, local12);

		cl::Kernel kernel3 = get_kernel(context, "batchnorm_backward");
		cl::NDRange global3(32 * ((last_dim + 31) / 32), 8 * std::min(1024, (first_dim + 7) / 8));
		cl::NDRange local3(32, 8);

		kernel3.setArg(0, workspace);
		kernel3.setArg(1, opencl::getMemoryObject(input).buffer());
		kernel3.setArg(2, opencl::getMemoryObject(gradient_prev).buffer());
		kernel3.setArg(3, opencl::getMemoryObject(gradient_next).buffer());
		kernel3.setArg(4, opencl::getMemoryObject(weights).buffer());
		kernel3.setArg(5, opencl::getMemoryObject(weights_update).buffer());
		kernel3.setArg(6, opencl::getMemoryObject(running_stats).buffer());
		kernel3.setArg(7, running_stat_idx);
		kernel3.setArg(8, first_dim);
		kernel3.setArg(9, last_dim);

		opencl::runKernel(context, kernel3, global3, local3);
	}
	void opencl_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape) / 3;

		cl::Kernel kernel = get_kernel(context, "batchnorm_update");
		cl::NDRange global(256 * std::max(1, (last_dim + 255) / 256));
		cl::NDRange local(256);

		kernel.setArg(0, opencl::getMemoryObject(running_stat).buffer());
		kernel.setArg(1, opencl::getMemoryObject(weights).buffer());
		kernel.setArg(2, first_dim);
		kernel.setArg(3, last_dim);
		kernel.setArg(4, static_cast<int>(use_gamma));
		kernel.setArg(5, static_cast<int>(use_beta));

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = volume_without_first_dim(shape);

		cl::Kernel kernel = get_kernel(context, "fold_batchnorm");
		cl::NDRange global(first_dim * 256);
		cl::NDRange local(256);

		kernel.setArg(0, first_dim);
		kernel.setArg(1, last_dim);
		kernel.setArg(2, opencl::getMemoryObject(layer_weights).buffer());
		kernel.setArg(3, opencl::getMemoryObject(layer_bias).buffer());
		kernel.setArg(4, opencl::getMemoryObject(batchnorm_weights).buffer());

		opencl::runKernel(context, kernel, global, local);
	}

	void opencl_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights,
			const void *bias, const void *ext)
	{
	}
	void opencl_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev,
			void *gradient_next, const void *weights, void *weights_update, void *bias_update)
	{
	}

} /* namespace ml */

