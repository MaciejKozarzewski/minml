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
	const cl::Program& get_program()
	{
		static const cl::Program program = ml::opencl::compile_program("training",
				ml::opencl::kernels::common + ml::opencl::kernels::reductions + ml::opencl::kernels::training, "");
		return program;

	}
}

namespace ml
{
	void opencl_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src)
	{
		const int elements = volume(shape);

		cl::Kernel kernel(get_program(), "emulate_low_precision");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::get_buffer(dst));
		kernel.setArg(1, opencl::get_buffer(src));
		kernel.setArg(2, elements);

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		const int elements = volume(shape);
		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		switch (dtype)
		{
//				case DTYPE_FLOAT16:
//					break;
			case DTYPE_FLOAT32:
				kernel = cl::Kernel(get_program(), "add_tensors");
				break;
			default:
				break;
		}

		kernel.setArg(0, opencl::get_buffer(dst));
		kernel.setArg(1, opencl::get_buffer(src1));
		kernel.setArg(2, opencl::get_buffer(src2));
		kernel.setArg(3, elements);

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta)
	{
		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		assert(opencl::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		const int workspace_first_dim = std::min((size_t) 256, opencl::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		cl::Kernel kernel(get_program(), "sum_over_first_dim");
		cl::NDRange local(32, 32);

		{ /* artificial scope for step 1 */
			cl::NDRange global(32 * ((last_dim + 31) / 32), 32 * workspace_first_dim);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::get_buffer(src));
			kernel.setArg(2, first_dim);
			kernel.setArg(3, last_dim);
			kernel.setArg(4, beta);
			kernel.setArg(5, 1);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}

		{ /* artificial scope for step 2 */
			cl::NDRange global(32 * ((last_dim + 31) / 32), 32);

			kernel.setArg(0, opencl::get_buffer(dst));
			kernel.setArg(1, workspace);
			kernel.setArg(2, workspace_first_dim);
			kernel.setArg(3, last_dim);
			kernel.setArg(4, beta);
			kernel.setArg(5, 2);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}
	}
	float opencl_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		cl::Buffer &workspace = opencl::Context::getWorkspace(context);
		assert(opencl::Context::getWorkspaceSize(context) >= 1024 * sizeof(float));
		const int elements = volume(shape);
		{ /* artificial scope for step 1 */
			cl::Kernel kernel(get_program(), "MSE_loss_step1");
			cl::NDRange global(256 * std::min(4096, ((elements + 255) / 256)));
			cl::NDRange local(256);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::get_buffer(output));
			kernel.setArg(2, opencl::get_buffer(target));
			kernel.setArg(3, elements);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}

		{ /* artificial scope for step 2 */
			cl::Kernel kernel(get_program(), "reduce_loss_step2");
			cl::NDRange global = opencl::get_nd_range<256>(elements);
			cl::NDRange local = opencl::get_nd_range<256>(elements);

			kernel.setArg(0, workspace);
			kernel.setArg(1, elements);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}

		float result = 0.0f;
		cl_int status = opencl::Context::getCommandQueue(context).enqueueReadBuffer(workspace, true, 0, sizeof(float), &result);
		assert(status == CL_SUCCESS);
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
			cl::Kernel kernel(get_program(), "CE_loss_step1");
			cl::NDRange global(256 * std::min(4096, ((elements + 255) / 256)));
			cl::NDRange local(256);

			kernel.setArg(0, workspace);
			kernel.setArg(1, opencl::get_buffer(output));
			kernel.setArg(2, opencl::get_buffer(target));
			kernel.setArg(3, elements);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}

		{ /* artificial scope for step 2 */
			cl::Kernel kernel(get_program(), "reduce_loss_step2");
			cl::NDRange global = opencl::get_nd_range<256>(elements);
			cl::NDRange local = opencl::get_nd_range<256>(elements);

			kernel.setArg(0, workspace);
			kernel.setArg(1, elements);

			cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			assert(status == CL_SUCCESS);
		}

		float result = 0.0f;
		cl_int status = opencl::Context::getCommandQueue(context).enqueueReadBuffer(workspace, true, 0, sizeof(float), &result);
		assert(status == CL_SUCCESS);
		return result / get_first_dim(shape);
	}
	void opencl_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		const int elements = volume(shape);
		cl::Kernel kernel(get_program(), "loss_gradient");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::get_buffer(gradient));
		kernel.setArg(1, opencl::get_buffer(output));
		kernel.setArg(2, opencl::get_buffer(target));
		kernel.setArg(3, elements);
		kernel.setArg(4, weight / get_first_dim(shape));

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_adam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2)
	{
		const int elements = volume(shape);
		cl::Kernel kernel(get_program(), "learn_adam");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::get_buffer(weight));
		kernel.setArg(1, opencl::get_buffer(update));
		kernel.setArg(2, opencl::get_buffer(momentum));
		kernel.setArg(3, opencl::get_buffer(variance));
		kernel.setArg(4, elements);
		kernel.setArg(5, learning_rate);
		kernel.setArg(6, beta1);
		kernel.setArg(7, beta2);

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
	void opencl_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset)
	{
		const int elements = volume(shape);
		cl::Kernel kernel(get_program(), "regularizer_l2");
		cl::NDRange global = opencl::get_nd_range<65536>(elements);
		cl::NDRange local;

		kernel.setArg(0, opencl::get_buffer(gradient));
		kernel.setArg(1, opencl::get_buffer(param));
		kernel.setArg(2, coefficient);
		kernel.setArg(3, offset);
		kernel.setArg(4, elements);

		cl_int status = opencl::Context::getCommandQueue(context).enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		assert(status == CL_SUCCESS);
	}
} /* namespace ml */

