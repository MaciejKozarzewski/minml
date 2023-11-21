/*
 * ml_memory.hpp
 *
 *  Created on: Oct 15, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_ML_MEMORY_HPP_
#define MINML_CORE_ML_MEMORY_HPP_

#include <cstddef>

namespace ml /* forward declarations */
{
	class Device;
	class Context;
}

namespace ml
{
	void* malloc(Device device, size_t count);
	void free(Device device, void *ptr);

	void* create_view(Device device, void *src, size_t offset, size_t count);
	void destroy_view(Device device, void *ptr);

	void memzero(Device dst_device, void *dst, size_t dst_offset, size_t dst_count);
	void memset(Device dst_device, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count);
	void memcpy(Device dst_device, void *dst_ptr, size_t dst_offset, Device src_device, const void *src_ptr, size_t src_offset, size_t count);

	void memzero_async(const Context &context, Device dst_device, void *dst, size_t dst_offset, size_t dst_count);
	void memset_async(const Context &context, Device dst_device, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count);
	void memcpy_async(const Context &context, Device dst_device, void *dst_ptr, size_t dst_offset, Device src_device, const void *src_ptr,
			size_t src_offset, size_t count);

} /* namespace ml */

#endif /* MINML_CORE_ML_MEMORY_HPP_ */
