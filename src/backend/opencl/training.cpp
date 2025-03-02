/*
 * training.cpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>

namespace
{
	cl::Kernel get_kernel(ml::mlContext_t context, const char *name)
	{
		static const ml::opencl::ProgramCache result("training",
				ml::opencl::kernels::common + ml::opencl::kernels::reductions + ml::opencl::kernels::training, "");
		return result.getKernel(context, name);
	}
}

namespace ml
{
	void opencl_emulate_low_precision(mlContext_t context, mlShape_t shape, mlDataType_t dtype, void *dst, const void *src, mlQuantizationData_t qd)
	{
		const int elements = volume(shape);

		cl::Kernel kernel = get_kernel(context, "emulate_low_precision");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::getMemoryObject(dst).buffer());
		kernel.setArg(1, opencl::getMemoryObject(src).buffer());
		kernel.setArg(2, elements);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		const int elements = volume(shape);
		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		switch (dtype)
		{
//			case DTYPE_FLOAT16:
//				break;
			case DTYPE_FLOAT32:
				kernel = get_kernel(context, "multiply_tensors");
				break;
			default:
				break;
		}

		kernel.setArg(0, opencl::getMemoryObject(dst).buffer());
		kernel.setArg(1, opencl::getMemoryObject(src1).buffer());
		kernel.setArg(2, opencl::getMemoryObject(src2).buffer());
		kernel.setArg(3, elements);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, float alpha1, const void *src1, float alpha2,
			const void *src2)
	{
		const int elements = volume(shape);
		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		switch (dtype)
		{
//			case DTYPE_FLOAT16:
//				break;
			case DTYPE_FLOAT32:
				kernel = get_kernel(context, "add_tensors");
				break;
			default:
				break;
		}

		kernel.setArg(0, opencl::getMemoryObject(dst).buffer());
		kernel.setArg(1, alpha1);
		kernel.setArg(2, opencl::getMemoryObject(src1).buffer());
		kernel.setArg(3, alpha2);
		kernel.setArg(4, opencl::getMemoryObject(src2).buffer());
		kernel.setArg(5, elements);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta)
	{
		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		assert(opencl::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		const int workspace_first_dim = std::min((size_t) 256, opencl::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		cl::Kernel kernel = get_kernel(context, "sum_over_first_dim");
		cl::NDRange local(32, 32);

		{ /* artificial scope for step 1 */
			cl::NDRange global(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::getMemoryObject(src).buffer());
			kernel.setArg(2, first_dim);
			kernel.setArg(3, last_dim);
			kernel.setArg(4, beta);
			kernel.setArg(5, 1);

			opencl::runKernel(context, kernel, global, local);
		}

		{ /* artificial scope for step 2 */
			cl::NDRange global(32 * ((last_dim + 31) / 32), 32);

			kernel.setArg(0, opencl::getMemoryObject(dst).buffer());
			kernel.setArg(1, workspace);
			kernel.setArg(2, workspace_first_dim);
			kernel.setArg(3, last_dim);
			kernel.setArg(4, beta);
			kernel.setArg(5, 2);

			opencl::runKernel(context, kernel, global, local);
		}
	}
	float opencl_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		assert(opencl::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));
		const int elements = volume(shape);
		{ /* artificial scope for step 1 */
			cl::Kernel kernel = get_kernel(context, "MSE_loss_step1");
			cl::NDRange global(256 * std::min(4096, ((elements + 255) / 256)));
			cl::NDRange local(256);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::getMemoryObject(output).buffer());
			kernel.setArg(2, opencl::getMemoryObject(target).buffer());
			kernel.setArg(3, elements);

			opencl::runKernel(context, kernel, global, local);
		}

		{ /* artificial scope for step 2 */
			cl::Kernel kernel = get_kernel(context, "reduce_loss_step2");
			cl::NDRange global = opencl::get_nd_range<256>(elements);
			cl::NDRange local = opencl::get_nd_range<256>(elements);

			kernel.setArg(0, workspace);
			kernel.setArg(1, elements);

			opencl::runKernel(context, kernel, global, local);
		}

		float result = 0.0f;
		opencl_memcpy_to_host(context, &result, &workspace, 0, sizeof(float));
		opencl_synchronize_with_context(context);
		return result / get_first_dim(shape);
	}
	void opencl_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		opencl_cross_entropy_gradient(context, shape, gradient, output, target, weight); // in this case both gradients are the same
	}
	float opencl_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		assert(opencl::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));
		const int elements = volume(shape);
		{ /* artificial scope for step 1 */
			cl::Kernel kernel = get_kernel(context, "CE_loss_step1");
			cl::NDRange global(256 * std::min(4096, ((elements + 255) / 256)));
			cl::NDRange local(256);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::getMemoryObject(output).buffer());
			kernel.setArg(2, opencl::getMemoryObject(target).buffer());
			kernel.setArg(3, elements);

			opencl::runKernel(context, kernel, global, local);
		}

		{ /* artificial scope for step 2 */
			cl::Kernel kernel = get_kernel(context, "reduce_loss_step2");
			cl::NDRange global = opencl::get_nd_range<256>(elements);
			cl::NDRange local = opencl::get_nd_range<256>(elements);

			kernel.setArg(0, workspace);
			kernel.setArg(1, elements);

			opencl::runKernel(context, kernel, global, local);
		}

		float result = 0.0f;
		opencl_memcpy_to_host(context, &result, &workspace, 0, sizeof(float));
		opencl_synchronize_with_context(context);
		return result / get_first_dim(shape);
	}
	void opencl_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		const int elements = volume(shape);
		cl::Kernel kernel = get_kernel(context, "loss_gradient");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::getMemoryObject(gradient).buffer());
		kernel.setArg(1, opencl::getMemoryObject(output).buffer());
		kernel.setArg(2, opencl::getMemoryObject(target).buffer());
		kernel.setArg(3, elements);
		kernel.setArg(4, weight / get_first_dim(shape));

		opencl::runKernel(context, kernel, global, local);
	}
	float opencl_value_head_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
	}
	void opencl_value_head_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
	}

	void opencl_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2, int step, float weight_decay)
	{
		const int elements = volume(shape);
		cl::Kernel kernel = get_kernel(context, "learn_adam");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::getMemoryObject(weight).buffer());
		kernel.setArg(1, opencl::getMemoryObject(update).buffer());
		kernel.setArg(2, opencl::getMemoryObject(momentum).buffer());
		kernel.setArg(3, opencl::getMemoryObject(variance).buffer());
		kernel.setArg(4, elements);
		kernel.setArg(5, learning_rate);
		kernel.setArg(6, beta1);
		kernel.setArg(7, beta2);
		kernel.setArg(8, step);

		opencl::runKernel(context, kernel, global, local);
	}

	void opencl_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset)
	{
		const int elements = volume(shape);
		cl::Kernel kernel = get_kernel(context, "regularizer_l2");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::getMemoryObject(gradient).buffer());
		kernel.setArg(1, opencl::getMemoryObject(param).buffer());
		kernel.setArg(2, coefficient);
		kernel.setArg(3, offset);
		kernel.setArg(4, elements);

		opencl::runKernel(context, kernel, global, local);
	}
} /* namespace ml */

