/*
 * opencl_properties.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>

#include "utils.hpp"

#include <CL/opencl.hpp>

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <bitset>
#include <iostream>

namespace
{
	struct openclDeviceProp
	{
			cl_device_type type;
			cl_uint vendor_id;
			cl_uint max_compute_units;
			cl_uint max_work_item_dimensions;
			std::vector<size_t> max_work_item_sizes;
			size_t max_work_group_size;

			cl_uint preferred_vector_width_char;
			cl_uint preferred_vector_width_short;
			cl_uint preferred_vector_width_int;
			cl_uint preferred_vector_width_long;
			cl_uint preferred_vector_width_half;
			cl_uint preferred_vector_width_float;
			cl_uint preferred_vector_width_double;

			cl_uint native_vector_width_char;
			cl_uint native_vector_width_short;
			cl_uint native_vector_width_int;
			cl_uint native_vector_width_long;
			cl_uint native_vector_width_half;
			cl_uint native_vector_width_float;
			cl_uint native_vector_width_double;

			cl_uint max_clock_frequency;
			cl_uint address_bits;
			cl_ulong max_mem_alloc_size;

			cl_bool image_support;
			cl_uint max_read_image_args;
			cl_uint max_write_image_args;
			cl_uint max_read_write_image_args;
			size_t image2D_max_width;
			size_t image2D_max_height;
			size_t image3D_max_width;
			size_t image3D_max_height;
			size_t image3D_max_depth;
			size_t image_max_buffer_size;
			size_t image_max_array_size;
			cl_uint max_samplers;
			cl_uint image_pitch_alignment;
			cl_uint image_base_address_alignment;

			cl_uint max_pipe_args;
			cl_uint pipe_max_active_reservations;
			cl_uint pipe_max_packet_size;

			size_t max_parameter_size;

			cl_uint mem_base_add_align;

			cl_device_fp_config single_fp_config;
			cl_device_fp_config double_fp_config;

			cl_device_mem_cache_type global_mem_cache_type;
			cl_uint global_mem_cacheline_size;
			cl_ulong global_mem_cache_size;
			cl_ulong global_mem_size;

			cl_ulong max_constant_buffer_size;
			cl_uint max_constant_args;

			size_t max_global_variable_size;
			size_t global_variable_preferred_total_size;

			cl_device_local_mem_type local_mem_type;
			cl_ulong local_mem_size;

			cl_bool error_correction_support;

			size_t profiling_timer_resolution;
			cl_bool endian_little;
			cl_bool available;

			cl_bool compiler_available;
			cl_bool linker_available;

			cl_device_exec_capabilities execution_capabilities;
			cl_command_queue_properties queue_on_host_properties;
			cl_command_queue_properties queue_on_device_properties;
			cl_uint queue_on_device_max_size;
			cl_uint max_on_device_queues;
			cl_uint max_on_device_events;

			std::string built_in_kernels;
			cl_platform_id platform;
			std::string name;
			std::string vendor;
			std::string driver_version;
			std::string profile;
			std::string version;
			std::string opencl_c_version;
			std::string extensions;

			size_t printf_buffer_size;
			cl_bool preferred_interop_user_sync;
			cl_device_id parent_device;
			cl_uint partition_max_sub_devices;
			std::vector<cl_device_partition_property> partition_properties;
			cl_device_affinity_domain partition_affinity_domain;
			std::vector<cl_device_partition_property> partition_type;

			cl_uint reference_count;

			cl_device_svm_capabilities svm_capabilities;
			cl_uint preferred_platform_atomic_alignment;
			cl_uint preferred_global_atomic_alignment;
			cl_uint preferred_local_atomic_alignment;

			openclDeviceProp(int index)
			{
				const cl::Device &device = ml::opencl::get_list_of_devices().at(index);
				device.getInfo(CL_DEVICE_TYPE, &type);
				device.getInfo(CL_DEVICE_VENDOR_ID, &vendor_id);
				device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units);
				device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &max_work_item_dimensions);
				device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
				device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &preferred_vector_width_char);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &preferred_vector_width_short);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &preferred_vector_width_int);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &preferred_vector_width_long);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &preferred_vector_width_half);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &preferred_vector_width_float);
				device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &preferred_vector_width_double);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &native_vector_width_char);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &native_vector_width_short);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &native_vector_width_int);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &native_vector_width_long);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &native_vector_width_half);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &native_vector_width_float);
				device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &native_vector_width_double);
				device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_clock_frequency);
				device.getInfo(CL_DEVICE_ADDRESS_BITS, &address_bits);
				device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
				device.getInfo(CL_DEVICE_IMAGE_SUPPORT, &image_support);
				device.getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &max_read_image_args);
				device.getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &max_write_image_args);
				device.getInfo(CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, &max_read_write_image_args);
				device.getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &image2D_max_height);
				device.getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &image2D_max_width);
				device.getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &image3D_max_height);
				device.getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &image3D_max_width);
				device.getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &image3D_max_depth);
				device.getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &image_max_buffer_size);
				device.getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &image_max_array_size);
				device.getInfo(CL_DEVICE_MAX_SAMPLERS, &max_samplers);
				device.getInfo(CL_DEVICE_IMAGE_PITCH_ALIGNMENT, &image_pitch_alignment);
				device.getInfo(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, &image_base_address_alignment);
				device.getInfo(CL_DEVICE_MAX_PIPE_ARGS, &max_pipe_args);
				device.getInfo(CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, &pipe_max_active_reservations);
				device.getInfo(CL_DEVICE_PIPE_MAX_PACKET_SIZE, &pipe_max_packet_size);
				device.getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &max_parameter_size);
				device.getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &mem_base_add_align);
				device.getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &single_fp_config);
				device.getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &double_fp_config);
				device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &global_mem_cache_type);
				device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &global_mem_cacheline_size);
				device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &global_mem_cache_size);
				device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
				device.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &max_constant_buffer_size);
				device.getInfo(CL_DEVICE_MAX_CONSTANT_ARGS, &max_constant_args);
				device.getInfo(CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, &max_global_variable_size);
				device.getInfo(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, &global_variable_preferred_total_size);
				device.getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &local_mem_type);
				device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
				device.getInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT, &error_correction_support);
				device.getInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &profiling_timer_resolution);
				device.getInfo(CL_DEVICE_ENDIAN_LITTLE, &endian_little);
				device.getInfo(CL_DEVICE_AVAILABLE, &available);
				device.getInfo(CL_DEVICE_COMPILER_AVAILABLE, &compiler_available);
				device.getInfo(CL_DEVICE_LINKER_AVAILABLE, &linker_available);
				device.getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &execution_capabilities);
				device.getInfo(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, &queue_on_host_properties);
				device.getInfo(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, &queue_on_device_properties);
				device.getInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, &queue_on_device_max_size);
				device.getInfo(CL_DEVICE_MAX_ON_DEVICE_QUEUES, &max_on_device_queues);
				device.getInfo(CL_DEVICE_MAX_ON_DEVICE_EVENTS, &max_on_device_events);
				device.getInfo(CL_DEVICE_BUILT_IN_KERNELS, &built_in_kernels);
				device.getInfo(CL_DEVICE_PLATFORM, &platform);
				device.getInfo(CL_DEVICE_NAME, &name);
				device.getInfo(CL_DEVICE_VENDOR, &vendor);
				device.getInfo(CL_DRIVER_VERSION, &driver_version);
				device.getInfo(CL_DEVICE_PROFILE, &profile);
				device.getInfo(CL_DEVICE_VERSION, &version);
				device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &opencl_c_version);
				device.getInfo(CL_DEVICE_EXTENSIONS, &extensions);
				device.getInfo(CL_DEVICE_PRINTF_BUFFER_SIZE, &printf_buffer_size);
				device.getInfo(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, &preferred_interop_user_sync);
				device.getInfo(CL_DEVICE_PARENT_DEVICE, &parent_device);
				device.getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &partition_max_sub_devices);
				device.getInfo(CL_DEVICE_PARTITION_PROPERTIES, &partition_properties);
				device.getInfo(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &partition_affinity_domain);
				device.getInfo(CL_DEVICE_PARTITION_TYPE, &partition_type);
				device.getInfo(CL_DEVICE_REFERENCE_COUNT, &reference_count);
				device.getInfo(CL_DEVICE_SVM_CAPABILITIES, &svm_capabilities);
				device.getInfo(CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, &preferred_platform_atomic_alignment);
				device.getInfo(CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, &preferred_global_atomic_alignment);
				device.getInfo(CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, &preferred_local_atomic_alignment);
			}
	};

	const std::vector<openclDeviceProp>& get_device_properties()
	{
		static const std::vector<openclDeviceProp> properties = []()
		{
			std::vector<openclDeviceProp> result;
			for (size_t i = 0; i < ml::opencl::get_list_of_devices().size(); i++)
				result.push_back(openclDeviceProp(i));
			return result;
		}();
		return properties;
	}
	std::vector<std::string> get_device_infos()
	{
		const std::vector<openclDeviceProp> &properties = get_device_properties();

		std::vector<std::string> result;
		for (auto prop = properties.begin(); prop < properties.end(); prop++)
		{
			std::string tmp = std::string(prop->name) + " : " + std::to_string(prop->max_compute_units) + " x " + prop->version;
			tmp += " with " + std::to_string(prop->global_mem_size >> 20) + "MB of memory";
			result.push_back(tmp);
		}
		return result;
	}

	template<typename T>
	void print_field(const std::string &name, const T &x)
	{
		std::cout << name << " : " << std::to_string(x) << '\n';
	}
	template<>
	void print_field<std::string>(const std::string &name, const std::string &x)
	{
		std::cout << name << " : \"" << x << "\"\n";
	}

	std::string to_hex(uint8_t x)
	{
		static const std::array<char, 16> text( { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' });
		return std::string(1, text[x / 16]) + std::string(1, text[x % 16]);
	}
	template<typename T>
	void print_hex(const std::string &name, T x)
	{
		std::cout << name << " : 0x";
		for (size_t i = 0; i < sizeof(T); i++)
			std::cout << to_hex(reinterpret_cast<const uint8_t*>(&x)[i]);
		std::cout << "\n";
	}
	template<typename T>
	void print_array(const std::string &name, const std::vector<T> &x)
	{
		std::cout << name << " : [";
		for (size_t i = 0; i < x.size(); i++)
			std::cout << ((i == 0) ? "" : ", ") << std::to_string(x[i]);
		std::cout << "]\n";
	}
	template<typename T>
	void print_bits(const std::string &name, T x)
	{
		std::cout << name << " : " << std::bitset<8 * sizeof(T)>(x).to_string() << '\n';
	}
}

namespace ml
{
	int opencl_get_number_of_devices()
	{
		static const int result = ml::opencl::get_list_of_devices().size();
		return result;
	}
	int opencl_get_memory(int index)
	{
		return get_device_properties().at(index).global_mem_size >> 20;
	}
	bool opencl_supports_type(int index, mlDataType_t dtype)
	{
		if (0 <= index && index < opencl_get_number_of_devices())
		{
			switch (dtype)
			{
				case DTYPE_FLOAT16:
				{
					static const bool result = get_device_properties().at(index).extensions.find("cl_khr_fp16") != std::string::npos;
					return result;
				}
				case DTYPE_FLOAT32:
					return true;
				default:
					return false;
			}
		}
		return false;
	}
	const char* opencl_get_device_info(int index)
	{
		static const std::vector<std::string> infos = get_device_infos();
		if (0 <= index && index < opencl_get_number_of_devices())
			return infos.at(index).data();
		else
			return nullptr;
	}
	void opencl_print_device_features(int index)
	{
		if (0 <= index && index < opencl_get_number_of_devices())
		{
			const openclDeviceProp &prop = get_device_properties().at(index);
			print_field("type", prop.type);
			print_field("vendor_id", prop.vendor_id);
			print_field("max_compute_units", prop.max_compute_units);
			print_field("max_work_item_dimensions", prop.max_work_item_dimensions);
			print_array("max_work_item_sizes", prop.max_work_item_sizes);
			print_field("max_work_group_size", prop.max_work_group_size);

			print_field("preferred_vector_width_char", prop.preferred_vector_width_char);
			print_field("preferred_vector_width_short", prop.preferred_vector_width_short);
			print_field("preferred_vector_width_int", prop.preferred_vector_width_int);
			print_field("preferred_vector_width_long", prop.preferred_vector_width_long);
			print_field("preferred_vector_width_half", prop.preferred_vector_width_half);
			print_field("preferred_vector_width_float", prop.preferred_vector_width_float);
			print_field("preferred_vector_width_double", prop.preferred_vector_width_double);

			print_field("native_vector_width_char", prop.native_vector_width_char);
			print_field("native_vector_width_short", prop.native_vector_width_short);
			print_field("native_vector_width_int", prop.native_vector_width_int);
			print_field("native_vector_width_long", prop.native_vector_width_long);
			print_field("native_vector_width_half", prop.native_vector_width_half);
			print_field("native_vector_width_float", prop.native_vector_width_float);
			print_field("native_vector_width_double", prop.native_vector_width_double);

			print_field("max_clock_frequency", prop.max_clock_frequency);
			print_field("address_bits", prop.address_bits);
			print_field("max_mem_alloc_size", prop.max_mem_alloc_size);

			print_field("image_support", prop.image_support);
			print_field("max_read_image_args", prop.max_read_image_args);
			print_field("max_write_image_args", prop.max_write_image_args);
			print_field("max_read_write_image_args", prop.max_read_write_image_args);
			print_field("image2D_max_width", prop.image2D_max_width);
			print_field("image2D_max_height", prop.image2D_max_height);
			print_field("image3D_max_width", prop.image3D_max_width);
			print_field("image3D_max_height", prop.image3D_max_height);
			print_field("image3D_max_depth", prop.image3D_max_depth);
			print_field("image_max_buffer_size", prop.image_max_buffer_size);
			print_field("image_max_array_size", prop.image_max_array_size);
			print_field("max_samplers", prop.max_samplers);
			print_field("image_pitch_alignment", prop.image_pitch_alignment);
			print_field("image_base_address_alignment", prop.image_base_address_alignment);

			print_field("max_pipe_args", prop.max_pipe_args);
			print_field("pipe_max_active_reservations", prop.pipe_max_active_reservations);
			print_field("pipe_max_packet_size", prop.pipe_max_packet_size);

			print_field("max_parameter_size", prop.max_parameter_size);
			print_field("mem_base_add_align", prop.mem_base_add_align);

			print_field("single_fp_config", prop.single_fp_config);
			print_field("double_fp_config", prop.double_fp_config);

			print_field("global_mem_cache_type", prop.global_mem_cache_type);
			print_field("global_mem_cacheline_size", prop.global_mem_cacheline_size);
			print_field("global_mem_cache_size", prop.global_mem_cache_size);
			print_field("global_mem_size", prop.global_mem_size);

			print_field("max_constant_buffer_size", prop.max_constant_buffer_size);
			print_field("max_constant_args", prop.max_constant_args);

			print_field("max_global_variable_size", prop.max_global_variable_size);
			print_field("global_variable_preferred_total_size", prop.global_variable_preferred_total_size);

			print_field("local_mem_type", prop.local_mem_type);
			print_field("local_mem_size", prop.local_mem_size);

			print_field("error_correction_support", prop.error_correction_support);

			print_field("profiling_timer_resolution", prop.profiling_timer_resolution);
			print_field("endian_little", prop.endian_little);
			print_field("available", prop.available);

			print_field("compiler_available", prop.compiler_available);
			print_field("linker_available", prop.linker_available);

			print_field("execution_capabilities", prop.execution_capabilities);
			print_field("queue_on_host_properties", prop.queue_on_host_properties);
			print_field("queue_on_device_properties", prop.queue_on_device_properties);
			print_field("queue_on_device_max_size", prop.queue_on_device_max_size);
			print_field("max_on_device_queues", prop.max_on_device_queues);
			print_field("max_on_device_events", prop.max_on_device_events);

			print_field("built_in_kernels", prop.built_in_kernels);
			print_hex("platform", prop.platform);
			print_field("name", prop.name);
			print_field("vendor", prop.vendor);
			print_field("driver_version", prop.driver_version);
			print_field("profile", prop.profile);
			print_field("version", prop.version);
			print_field("opencl_c_version", prop.opencl_c_version);
			print_field("extensions", prop.extensions);

			print_field("printf_buffer_size", prop.printf_buffer_size);
			print_field("preferred_interop_user_sync", prop.preferred_interop_user_sync);
			print_hex("parent_device", prop.parent_device);
			print_field("partition_max_sub_devices", prop.partition_max_sub_devices);
			print_array("partition_properties", prop.partition_properties);
			print_field("partition_affinity_domain", prop.partition_affinity_domain);
			print_array("partition_type", prop.partition_type);

			print_field("reference_count", prop.reference_count);
			print_field("svm_capabilities", prop.svm_capabilities);
			print_field("preferred_platform_atomic_alignment", prop.preferred_platform_atomic_alignment);
			print_field("preferred_global_atomic_alignment", prop.preferred_global_atomic_alignment);
			print_field("preferred_local_atomic_alignment", prop.preferred_local_atomic_alignment);
		}
	}

} /* namespace ml */

