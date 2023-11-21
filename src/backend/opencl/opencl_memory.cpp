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
		if (count <= 0)
			return nullptr;

		cl_int status;
		cl::Buffer *result = new cl::Buffer(opencl::get_cl_context(), CL_MEM_READ_WRITE, count, nullptr, &status); // @suppress("Ambiguous problem")
		if (status != CL_BUILD_SUCCESS)
		{
			delete result;
			CHECK_OPENCL_STATUS(status);
		}
		return result;
	}
	void opencl_free(void *ptr)
	{
		if (ptr != nullptr)
			delete reinterpret_cast<cl::Buffer*>(ptr);
	}
	void* opencl_create_view(void *src, int offset, int count)
	{
		assert(src != nullptr);
		_cl_buffer_region tmp;
		tmp.origin = offset;
		tmp.size = count;

		cl_int status;
		cl::Buffer *result = new cl::Buffer();
		*result = opencl::getBuffer(src).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmp, &status);
		if (status != CL_SUCCESS)
		{
			delete result;
			CHECK_OPENCL_STATUS(status);
		}
		return result;
	}
	void opencl_destroy_view(void *ptr)
	{
		opencl_free(ptr);
	}

	void opencl_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		cl::Event local_event;
		cl::Event *event = (context == nullptr) ? &local_event : opencl::Context::getLastEvent(context);

		if (src == nullptr)
		{
			cl_int status = queue.enqueueFillBuffer<cl_char>(opencl::getBuffer(dst), 0u, dst_offset, dst_count, nullptr, event);
			CHECK_OPENCL_STATUS(status);
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
					cl_int status = queue.enqueueFillBuffer(opencl::getBuffer(dst), pattern, dst_offset, dst_count, nullptr, event);
					CHECK_OPENCL_STATUS(status);
					break;
				}
				case 4:
				{
					cl_int pattern;
					std::memcpy(&pattern, src, sizeof(cl_int));
					cl_int status = queue.enqueueFillBuffer(opencl::getBuffer(dst), pattern, dst_offset, dst_count, nullptr, event);
					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}
	void opencl_memcpy_within_device(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count)
	{
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		cl::Event local_event;
		cl::Event *event = (context == nullptr) ? &local_event : opencl::Context::getLastEvent(context);

		cl_int status = queue.enqueueCopyBuffer(opencl::getBuffer(src), opencl::getBuffer(dst), src_offset, dst_offset, count, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}
	void opencl_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count)
	{
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		cl::Event local_event;
		cl::Event *event = (context == nullptr) ? &local_event : opencl::Context::getLastEvent(context);

		const bool is_blocking = (context == nullptr);
		cl_int status = queue.enqueueWriteBuffer(opencl::getBuffer(dst), is_blocking, dst_offset, count, src, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}
	void opencl_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count)
	{
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		cl::Event local_event;
		cl::Event *event = (context == nullptr) ? &local_event : opencl::Context::getLastEvent(context);

		const bool is_blocking = (context == nullptr);
		cl_int status = queue.enqueueReadBuffer(opencl::getBuffer(src), is_blocking, src_offset, count, dst, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}

} /* namespace ml */

