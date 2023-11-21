/*
 * cpu_memory.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>

#include <cinttypes>
#include <cstring>
#include <algorithm>
#include <cassert>

namespace ml
{
	void* cpu_malloc(int count)
	{
		if (count <= 0)
			return nullptr;
		else
			return ::operator new[](count, std::align_val_t(64));
	}
	void cpu_free(void *ptr)
	{
		if (ptr != nullptr)
			::operator delete[](ptr, std::align_val_t(64));
	}
	void* cpu_create_view(void *src, int offset, int count)
	{
		if (src == nullptr)
			return nullptr;
		else
			return reinterpret_cast<uint8_t*>(src) + offset;
	}
	void cpu_destroy_view(void *ptr)
	{
	}

	void cpu_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count)
	{
		assert(dst != nullptr);
		if (src == nullptr)
			std::memset(reinterpret_cast<uint8_t*>(dst) + dst_offset, 0, dst_count);
		else
		{
			assert(dst_count % src_count == 0);

			uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(dst) + dst_offset;
			// buffer size must be divisible by pattern size, using about 256 bytes, but not less than then the actual pattern size
			const int buffer_size = src_count * std::max(1, (256 / src_count)); // bytes
			if (dst_count >= 4 * buffer_size)
			{
				uint8_t buffer[buffer_size];
				for (int i = 0; i < buffer_size; i += src_count)
					std::memcpy(buffer + i, src, src_count);
				for (int i = 0; i < dst_count; i += buffer_size)
					std::memcpy(dst_ptr + i, buffer, std::min(buffer_size, dst_count - i));
			}
			else
			{
				for (int i = 0; i < dst_count; i += src_count)
					std::memcpy(dst_ptr + i, src, src_count);
			}
		}
	}
	void cpu_memcpy(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		std::memcpy(reinterpret_cast<uint8_t*>(dst) + dst_offset, reinterpret_cast<const uint8_t*>(src) + src_offset, count);
	}
}

