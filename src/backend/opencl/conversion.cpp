/*
 * conversion.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>

namespace
{
	const cl::Program& get_program()
	{
		static const cl::Program program = ml::opencl::compileProgram("conversions", ml::opencl::kernels::conversion, "");
		return program;
	}
}

namespace ml
{
	void opencl_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		cl::Kernel kernel;

		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				kernel = opencl::getKernel(get_program(), "unpack_input_fp16");
				break;
			case DTYPE_FLOAT32:
				kernel = opencl::getKernel(get_program(), "unpack_input_fp32");
				break;
			case DTYPE_INT32:
			case DTYPE_UNKNOWN:
				break;
		}

		kernel.setArg(0, opencl::getBuffer(dst));
		kernel.setArg(1, opencl::getBuffer(src));
		kernel.setArg(2, first_dim);
		kernel.setArg(3, last_dim);

		const cl::NDRange global = opencl::get_nd_range<1024>(first_dim, 32);
		const cl::NDRange local = opencl::get_nd_range(1, 32);

		opencl::runKernel(context, kernel, global, local);
	}
	void opencl_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		if (dst_dtype == src_dtype and dst != src)
		{ // same type, different locations, can just copy memory
			ml::opencl_memcpy_within_device(context, dst, 0, src, 0, size_of(dst_dtype) * elements);
			return;
		}

		cl::Kernel kernel;
		if (dst_dtype == DTYPE_FLOAT16 and src_dtype == DTYPE_FLOAT32)
		{
			assert(dst != src);
			kernel = opencl::getKernel(get_program(), "convert_fp32_to_fp16");
		}
		if (dst_dtype == DTYPE_FLOAT32 and src_dtype == DTYPE_FLOAT16)
		{
			assert(dst != src);
			kernel = opencl::getKernel(get_program(), "convert_fp16_to_fp32");
		}

		kernel.setArg(0, opencl::getBuffer(dst));
		kernel.setArg(1, opencl::getBuffer(src));
		kernel.setArg(2, elements);

		const cl::NDRange global = opencl::get_nd_range<65536>(elements);

		opencl::runKernel(context, kernel, global);
	}

	void opencl_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
//		assert(input != output);
//
//		dim3 blockDim(64, 8);
//		cudaStream_t stream = cuda::Context::getStream(context);
//
//		switch (dtype)
//		{
//			case DTYPE_FLOAT16:
//			{
//				dim3 gridDim(shape.dim[0], (shape.dim[1] + 128 - 1) / 128, (shape.dim[2] + 64 - 1) / 64);
//				kernel_transpose_021<uint16_t, 128, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint16_t>(output), getPointer<uint16_t>(input),
//						shape.dim[0], shape.dim[1], shape.dim[2]);
//				break;
//			}
//			case DTYPE_FLOAT32:
//			case DTYPE_INT32:
//			{
//				dim3 gridDim(shape.dim[0], (shape.dim[1] + 64 - 1) / 64, (shape.dim[2] + 64 - 1) / 64);
//				kernel_transpose_021<uint32_t, 64, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<uint32_t>(output), getPointer<uint32_t>(input),
//						shape.dim[0], shape.dim[1], shape.dim[2]);
//				break;
//			}
//			default:
//				break;
//		}
//		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

