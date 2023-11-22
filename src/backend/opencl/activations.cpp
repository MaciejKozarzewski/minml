/*
 * activations.cpp
 *
 *  Created on: Nov 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>

namespace ml
{
	void opencl_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		static const cl::Program program = ml::opencl::compileProgram("activations_forward",
				ml::opencl::kernels::common + ml::opencl::kernels::reductions + ml::opencl::kernels::activations_forward, "");

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<65536>(volume(shape));
		cl::NDRange local = cl::NullRange;

		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (output != input)
					ml::opencl_memcpy_within_device(context, output, 0, input, 0, size_of(dtype) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
			{
				switch (dtype)
				{
//					case DTYPE_FLOAT16:
//						kernel = opencl::get_kernel(program, "sigmoid_forward_fp16");
//						break;
					case DTYPE_FLOAT32:
						kernel = opencl::getKernel(program, "sigmoid_forward_fp32");
						break;
					default:
						break;
				}
				break;
			}
			case ACTIVATION_TANH:
			{
				switch (dtype)
				{
//					case DTYPE_FLOAT16:
//						kernel = opencl::get_kernel(program, "tanh_forward_fp16");
//						break;
					case DTYPE_FLOAT32:
						kernel = opencl::getKernel(program, "tanh_forward_fp32");
						break;
					default:
						break;
				}
				break;
			}
			case ACTIVATION_RELU:
			{
				switch (dtype)
				{
//					case DTYPE_FLOAT16:
//						kernel = opencl::get_kernel(program, "relu_forward_fp16");
//						break;
					case DTYPE_FLOAT32:
						kernel = opencl::getKernel(program, "relu_forward_fp16");
						break;
					default:
						break;
				}
				break;
			}
			case ACTIVATION_SOFTMAX:
			{
				assert(shape.rank == 2);
				if (last_dim == 3)
				{
					global = opencl::get_nd_range(first_dim);
					switch (dtype)
					{
//						case DTYPE_FLOAT16:
//							kernel = opencl::get_kernel(program, "softmax_3_channels_fp16");
//							break;
						case DTYPE_FLOAT32:
							kernel = opencl::getKernel(program, "softmax_3_channels_fp32");
							break;
						default:
							break;
					}
				}
				else
				{
					global = opencl::get_nd_range<1024 * 128>(first_dim * 128);
					local = opencl::get_nd_range<128>(last_dim);
					switch (dtype)
					{
//						case DTYPE_FLOAT16:
//							kernel = opencl::get_kernel(program, "softmax_generic_fp16");
//							break;
						case DTYPE_FLOAT32:
							kernel = opencl::getKernel(program, "softmax_generic_fp32");
							break;
						default:
							break;
					}
				}
				break;
			}
		}

		kernel.setArg(0, opencl::getBuffer(output));
		kernel.setArg(1, opencl::getBuffer(input));
		kernel.setArg(2, first_dim);
		kernel.setArg(3, last_dim);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		static const cl::Program program = ml::opencl::compileProgram("activations_backward",
				ml::opencl::kernels::common + ml::opencl::kernels::activations_backward, "");
		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<65536>(volume(shape));
		cl::NDRange local = cl::NullRange;

		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (gradient_prev != gradient_next)
					ml::opencl_memcpy_within_device(context, gradient_prev, 0, gradient_next, 0, sizeof(float) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
				kernel = opencl::getKernel(program, "sigmoid_backward_fp32");
				break;
			case ACTIVATION_TANH:
				kernel = opencl::getKernel(program, "tanh_backward_fp32");
				break;
			case ACTIVATION_RELU:
				kernel = opencl::getKernel(program, "relu_backward_fp32");
				break;
			case ACTIVATION_SOFTMAX:
				break;
		}
		kernel.setArg(0, opencl::getBuffer(gradient_prev));
		kernel.setArg(1, opencl::getBuffer(gradient_next));
		kernel.setArg(2, opencl::getBuffer(output));
		kernel.setArg(3, volume(shape));

		opencl::runKernel(context, kernel, global, local);
	}

	void opencl_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias, mlActivationType_t act)
	{
		static const cl::Program program = ml::opencl::compileProgram("add_bias_act",
				ml::opencl::kernels::common + ml::opencl::kernels::add_bias_act, "");

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Kernel kernel;
		cl::NDRange global = opencl::get_nd_range<std::numeric_limits<int>::max(), 1024>(last_dim, first_dim);
		cl::NDRange local = opencl::get_nd_range<128, 1>(last_dim, 1);

		switch (dtype)
		{
//			case DTYPE_FLOAT16:
//				kernel = opencl::get_kernel(program, "sigmoid_forward_fp16");
//				break;
			case DTYPE_FLOAT32:
				kernel = opencl::getKernel(program, "add_bias_act_fp32");
				break;
			default:
				break;
		}

		kernel.setArg(0, opencl::getBuffer(output));
		kernel.setArg(1, opencl::getBuffer(input));
		kernel.setArg(2, opencl::getBuffer(bias));
		kernel.setArg(3, first_dim);
		kernel.setArg(4, last_dim);
		kernel.setArg(5, static_cast<int>(act));

		opencl::runKernel(context, kernel, global, local);
	}

} /* namespace ml */

