/*
 * device_memory.cpp
 *
 *  Created on: Oct 15, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/ml_memory.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>

#include <cstring>
#include <memory>

namespace
{
	using namespace ml;

	const void* apply_offset(const void *ptr, size_t offset) noexcept
	{
		if (ptr == nullptr)
			return nullptr;
		else
			return reinterpret_cast<const uint8_t*>(ptr) + offset;
	}
	void* apply_offset(void *ptr, size_t offset) noexcept
	{
		if (ptr == nullptr)
			return nullptr;
		else
			return reinterpret_cast<uint8_t*>(ptr) + offset;
	}

	void memset_impl(void *context, Device dst_device, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count)
	{
		if (dst == nullptr)
			return;
		switch (dst_device.type())
		{
			case DeviceType::CPU:
				cpu_memset(context, dst, dst_offset, dst_count, src, src_count);
				break;
			case DeviceType::CUDA:
				cuda_memset(context, dst, dst_offset, dst_count, src, src_count);
				break;
		}
	}
	void memcpy_impl(void *context, Device dst_device, void *dst_ptr, size_t dst_offset, Device src_device, const void *src_ptr, size_t src_offset,
			size_t count)
	{
		if (count == 0)
			return;

		switch (src_device.type())
		{
			case DeviceType::CPU:
			{
				switch (dst_device.type())
				{
					case DeviceType::CPU: // CPU -> CPU
						cpu_memcpy(context, dst_ptr, dst_offset, src_ptr, src_offset, count);
						break;
					case DeviceType::CUDA: // CPU -> CUDA
						cuda_memcpy_from_host(context, dst_ptr, dst_offset, apply_offset(src_ptr, src_offset), count);
						break;
				}
				break;
			}
			case DeviceType::CUDA:
			{
				switch (dst_device.type())
				{
					case DeviceType::CPU: // CUDA -> CPU
						cuda_memcpy_to_host(context, apply_offset(dst_ptr, dst_offset), src_ptr, src_offset, count);
						break;
					case DeviceType::CUDA: // CUDA -> CUDA
						std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(count);
						cuda_memcpy_to_host(context, buffer.get(), src_ptr, src_offset, count);
						cuda_memcpy_from_host(context, dst_ptr, dst_offset, buffer.get(), count);
						break;
				}
				break;
			}
		}
	}
}

namespace ml
{
	void* malloc(Device device, size_t count)
	{
		switch (device.type())
		{
			case DeviceType::CPU:
				return cpu_malloc(count);
			case DeviceType::CUDA:
				return cuda_malloc(device.index(), count);
			default:
				return nullptr;
		}
	}
	void* view(Device device, void *src, size_t offset, size_t count)
	{
		switch (device.type())
		{
			case DeviceType::CPU:
				return cpu_view(src, offset, count);
			case DeviceType::CUDA:
				return cuda_view(src, offset, count);
			default:
				return nullptr;
		}
	}
	void free(Device device, void *ptr)
	{
		switch (device.type())
		{
			case DeviceType::CPU:
				cpu_free(ptr);
				break;
			case DeviceType::CUDA:
				cuda_free(ptr);
				break;
		}
	}

	void memzero(Device dst_device, void *dst, size_t dst_offset, size_t dst_count)
	{
		memset_impl(nullptr, dst_device, dst, dst_offset, dst_count, nullptr, 0);
	}

	void memset(Device dst_device, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count)
	{
		memset_impl(nullptr, dst_device, dst, dst_offset, dst_count, src, src_count);
	}
	void memcpy(Device dst_device, void *dst_ptr, size_t dst_offset, Device src_device, const void *src_ptr, size_t src_offset, size_t count)
	{
		memcpy_impl(nullptr, dst_device, dst_ptr, dst_offset, src_device, src_ptr, src_offset, count);
	}

	void memzero_async(const Context &context, Device dst_device, void *dst, size_t dst_offset, size_t dst_count)
	{
		memset_impl(context.backend(), dst_device, dst, dst_offset, dst_count, nullptr, 0);
	}
	void memset_async(const Context &context, Device dst_device, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count)
	{
		memset_impl(context.backend(), dst_device, dst, dst_offset, dst_count, src, src_count);
	}
	void memcpy_async(const Context &context, Device dst_device, void *dst_ptr, size_t dst_offset, Device src_device, const void *src_ptr,
			size_t src_offset, size_t count)
	{
		memcpy_impl(context.backend(), dst_device, dst_ptr, dst_offset, src_device, src_ptr, src_offset, count);
	}

} /* namespace ml */
