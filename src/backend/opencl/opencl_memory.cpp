/*
 * opencl_memory.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>
#include "utils.hpp"

#include <CL/opencl.hpp>

#include <cinttypes>
#include <cstring>
#include <iostream>
#include <cassert>

namespace ml
{
	void* opencl_malloc(int device_index, int count)
	{
		void *result = nullptr;
		if (count > 0)
			return new cl::Buffer(opencl::get_cl_context(), CL_MEM_READ_WRITE, count);
		return result;
	}
	void opencl_free(void *ptr)
	{
		if (ptr != nullptr)
			delete reinterpret_cast<cl::Buffer*>(ptr);
	}
	void* opencl_view(void *src, int offset, int count)
	{
		if (src == nullptr)
			return nullptr;
		else
		{
			_cl_buffer_region tmp;
			tmp.origin = offset;
			tmp.size = count;

			cl_int status;
			cl::Buffer *result = new cl::Buffer();
			*result = opencl::get_buffer(src).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmp, &status);
			if (status != CL_SUCCESS)
				throw std::runtime_error("opencl_view() : could not create a sub-buffer (status = " + std::to_string(status));
			return result;
		}
	}

	void opencl_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		if (src == nullptr)
		{
			cl_int status = queue.enqueueFillBuffer<cl_char>(opencl::get_buffer(dst), 0u, dst_offset, dst_count);
			assert(status == CL_SUCCESS);
		}
		else
		{
			assert(dst_count % src_count == 0);
			switch (src_count)
			{
				case 2:
				{
					cl_short pattern;
					std::memcpy(&pattern, src, sizeof(cl_short));
					cl_int status = queue.enqueueFillBuffer(opencl::get_buffer(dst), pattern, dst_offset, dst_count);
					assert(status == CL_SUCCESS);
					break;
				}
				case 4:
				{
					cl_int pattern;
					std::memcpy(&pattern, src, sizeof(cl_int));
					cl_int status = queue.enqueueFillBuffer(opencl::get_buffer(dst), pattern, dst_offset, dst_count);
					assert(status == CL_SUCCESS);
					break;
				}
			}
		}
		if (opencl::Context::isSynchronized(context))
			opencl_synchronize_with_context(context);
	}
	void opencl_memcpy_within_device(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count)
	{
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		cl_int status = queue.enqueueCopyBuffer(opencl::get_buffer(src), opencl::get_buffer(dst), src_offset, dst_offset, count);
		assert(status == CL_SUCCESS);
		if (opencl::Context::isSynchronized(context))
			opencl_synchronize_with_context(context);
	}
	void opencl_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count)
	{
		assert(src != nullptr);
		const bool is_blocking = (context == nullptr);
		cl_int status = opencl::Context::getCommandQueue(context).enqueueWriteBuffer(opencl::get_buffer(dst), is_blocking, dst_offset, count, src);
		assert(status == CL_SUCCESS);
	}
	void opencl_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count)
	{
		assert(dst != nullptr);
		const bool is_blocking = (context == nullptr);
		cl_int status = opencl::Context::getCommandQueue(context).enqueueReadBuffer(opencl::get_buffer(src), is_blocking, src_offset, count, dst);
		assert(status == CL_SUCCESS);
	}

} /* namespace ml */

