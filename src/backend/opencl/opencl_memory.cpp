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
			return getPointer<uint8_t>(src) + offset;
	}

	void opencl_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		if (src == nullptr)
			queue.enqueueFillBuffer<cl_char>(opencl::get_buffer(dst), 0u, dst_offset, dst_count);
		else
		{
			assert(dst_count % src_count == 0);

			switch (src_count)
			{
				case 2:
				{
					cl_short pattern;
					std::memcpy(&pattern, src, sizeof(cl_short));
					queue.enqueueFillBuffer(opencl::get_buffer(dst), pattern, dst_offset, dst_count);
					break;
				}
				case 4:
				{
					cl_int pattern;
					std::memcpy(&pattern, src, sizeof(cl_int));
					queue.enqueueFillBuffer(opencl::get_buffer(dst), pattern, dst_offset, dst_count);
					break;
				}
			}
		}
		if (opencl::Context::isSynchronized(context))
			opencl_synchronize_with_context(context);
	}
	void opencl_memcpy_within_device(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		queue.enqueueCopyBuffer(opencl::get_buffer(src), opencl::get_buffer(dst), src_offset, dst_offset, count);
		if (opencl::Context::isSynchronized(context))
			opencl_synchronize_with_context(context);
	}
	void opencl_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const bool is_blocking = (context == nullptr);
		opencl::Context::getCommandQueue(context).enqueueWriteBuffer(opencl::get_buffer(dst), is_blocking, dst_offset, count, src);
	}
	void opencl_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const bool is_blocking = (context == nullptr);
		opencl::Context::getCommandQueue(context).enqueueReadBuffer(opencl::get_buffer(src), is_blocking, src_offset, count, dst);
	}

} /* namespace ml */

