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

		opencl::MemoryObject *result = nullptr;
		try
		{
			result = new opencl::MemoryObject(device_index, count);
		} catch (std::exception &e)
		{
			delete result;
			throw e;
		}
		return result;
	}
	void opencl_free(void *ptr)
	{
		if (ptr != nullptr)
			delete reinterpret_cast<opencl::MemoryObject*>(ptr);
	}
	void* opencl_create_view(void *src, int offset, int count)
	{
		assert(src != nullptr);
		return new opencl::MemoryObject(*reinterpret_cast<opencl::MemoryObject*>(src), offset, count);
	}
	void opencl_destroy_view(void *ptr)
	{
		opencl_free(ptr);
	}

	void opencl_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		cl::CommandQueue *queue;
		cl::Event *event;

		cl::Event local_event;
		if (context == nullptr)
		{
			queue = &opencl::Context::getDefaultCommandQueue(opencl::getMemoryObject(dst).getDeviceIndex());
			event = &local_event;
		}
		else
		{
			queue = &opencl::Context::getCommandQueue(context);
			event = opencl::Context::getLastEvent(context);
		}

		if (src == nullptr)
		{
			cl_int status = queue->enqueueFillBuffer<cl_char>(opencl::getMemoryObject(dst).buffer(), 0u, dst_offset, dst_count, nullptr, event);
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
					cl_int status = queue->enqueueFillBuffer(opencl::getMemoryObject(dst).buffer(), pattern, dst_offset, dst_count, nullptr, event);
					CHECK_OPENCL_STATUS(status);
					break;
				}
				case 4:
				{
					cl_int pattern;
					std::memcpy(&pattern, src, sizeof(cl_int));
					cl_int status = queue->enqueueFillBuffer(opencl::getMemoryObject(dst).buffer(), pattern, dst_offset, dst_count, nullptr, event);
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
		cl::CommandQueue *queue;
		cl::Event *event;

		cl::Event local_event;
		if (context == nullptr)
		{
			queue = &opencl::Context::getDefaultCommandQueue(opencl::getMemoryObject(src).getDeviceIndex());
			event = &local_event;
		}
		else
		{
			queue = &opencl::Context::getCommandQueue(context);
			event = opencl::Context::getLastEvent(context);
		}

		cl_int status = queue->enqueueCopyBuffer(opencl::getMemoryObject(src).buffer(), opencl::getMemoryObject(dst).buffer(), src_offset, dst_offset,
				count, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}
	void opencl_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count)
	{
		cl::CommandQueue *queue;
		cl::Event *event;

		cl::Event local_event;
		if (context == nullptr)
		{
			queue = &opencl::Context::getDefaultCommandQueue(opencl::getMemoryObject(dst).getDeviceIndex());
			event = &local_event;
		}
		else
		{
			queue = &opencl::Context::getCommandQueue(context);
			event = opencl::Context::getLastEvent(context);
		}

		const bool is_blocking = (context == nullptr);
		cl_int status = queue->enqueueWriteBuffer(opencl::getMemoryObject(dst).buffer(), is_blocking, dst_offset, count, src, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}
	void opencl_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count)
	{
		cl::CommandQueue *queue;
		cl::Event *event;

		cl::Event local_event;
		if (context == nullptr)
		{
			queue = &opencl::Context::getDefaultCommandQueue(opencl::getMemoryObject(src).getDeviceIndex());
			event = &local_event;
		}
		else
		{
			queue = &opencl::Context::getCommandQueue(context);
			event = opencl::Context::getLastEvent(context);
		}

		const bool is_blocking = (context == nullptr);
		cl_int status = queue->enqueueReadBuffer(opencl::getMemoryObject(src).buffer(), is_blocking, src_offset, count, dst, nullptr, event);
		CHECK_OPENCL_STATUS(status);
		if (opencl::Context::isSynchronized(context))
			opencl::waitForEvent(event);
	}

} /* namespace ml */

