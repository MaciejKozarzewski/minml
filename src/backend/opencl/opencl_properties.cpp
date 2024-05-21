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
	std::string to_hex(uint8_t x)
	{
		static const std::array<char, 16> text( { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' });
		return std::string(1, text[x / 16]) + std::string(1, text[x % 16]);
	}

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
				get_info(device, CL_DEVICE_TYPE, &type);
				get_info(device, CL_DEVICE_VENDOR_ID, &vendor_id);
				get_info(device, CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units);
				get_info(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &max_work_item_dimensions);
				get_info(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
				get_info(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &preferred_vector_width_char);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &preferred_vector_width_short);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &preferred_vector_width_int);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &preferred_vector_width_long);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &preferred_vector_width_half);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &preferred_vector_width_float);
				get_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &preferred_vector_width_double);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &native_vector_width_char);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &native_vector_width_short);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &native_vector_width_int);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &native_vector_width_long);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &native_vector_width_half);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &native_vector_width_float);
				get_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &native_vector_width_double);
				get_info(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_clock_frequency);
				get_info(device, CL_DEVICE_ADDRESS_BITS, &address_bits);
				get_info(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
				get_info(device, CL_DEVICE_IMAGE_SUPPORT, &image_support);
				get_info(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, &max_read_image_args);
				get_info(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &max_write_image_args);
				get_info(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, &max_read_write_image_args);
				get_info(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, &image2D_max_height);
				get_info(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, &image2D_max_width);
				get_info(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, &image3D_max_height);
				get_info(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, &image3D_max_width);
				get_info(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, &image3D_max_depth);
				get_info(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &image_max_buffer_size);
				get_info(device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &image_max_array_size);
				get_info(device, CL_DEVICE_MAX_SAMPLERS, &max_samplers);
				get_info(device, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, &image_pitch_alignment);
				get_info(device, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, &image_base_address_alignment);
				get_info(device, CL_DEVICE_MAX_PIPE_ARGS, &max_pipe_args);
				get_info(device, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, &pipe_max_active_reservations);
				get_info(device, CL_DEVICE_PIPE_MAX_PACKET_SIZE, &pipe_max_packet_size);
				get_info(device, CL_DEVICE_MAX_PARAMETER_SIZE, &max_parameter_size);
				get_info(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, &mem_base_add_align);
				get_info(device, CL_DEVICE_SINGLE_FP_CONFIG, &single_fp_config);
				get_info(device, CL_DEVICE_DOUBLE_FP_CONFIG, &double_fp_config);
				get_info(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &global_mem_cache_type);
				get_info(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &global_mem_cacheline_size);
				get_info(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &global_mem_cache_size);
				get_info(device, CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
				get_info(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &max_constant_buffer_size);
				get_info(device, CL_DEVICE_MAX_CONSTANT_ARGS, &max_constant_args);
				get_info(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, &max_global_variable_size);
				get_info(device, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, &global_variable_preferred_total_size);
				get_info(device, CL_DEVICE_LOCAL_MEM_TYPE, &local_mem_type);
				get_info(device, CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
				get_info(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, &error_correction_support);
				get_info(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, &profiling_timer_resolution);
				get_info(device, CL_DEVICE_ENDIAN_LITTLE, &endian_little);
				get_info(device, CL_DEVICE_AVAILABLE, &available);
				get_info(device, CL_DEVICE_COMPILER_AVAILABLE, &compiler_available);
				get_info(device, CL_DEVICE_LINKER_AVAILABLE, &linker_available);
				get_info(device, CL_DEVICE_EXECUTION_CAPABILITIES, &execution_capabilities);
				get_info(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, &queue_on_host_properties);
				get_info(device, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, &queue_on_device_properties);
				get_info(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, &queue_on_device_max_size);
				get_info(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, &max_on_device_queues);
				get_info(device, CL_DEVICE_MAX_ON_DEVICE_EVENTS, &max_on_device_events);
				get_info(device, CL_DEVICE_BUILT_IN_KERNELS, &built_in_kernels);
				get_info(device, CL_DEVICE_PLATFORM, &platform);
				get_info(device, CL_DEVICE_NAME, &name);
				get_info(device, CL_DEVICE_VENDOR, &vendor);
				get_info(device, CL_DRIVER_VERSION, &driver_version);
				get_info(device, CL_DEVICE_PROFILE, &profile);
				get_info(device, CL_DEVICE_VERSION, &version);
				get_info(device, CL_DEVICE_OPENCL_C_VERSION, &opencl_c_version);
				get_info(device, CL_DEVICE_EXTENSIONS, &extensions);
				get_info(device, CL_DEVICE_PRINTF_BUFFER_SIZE, &printf_buffer_size);
				get_info(device, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, &preferred_interop_user_sync);
				get_info(device, CL_DEVICE_PARENT_DEVICE, &parent_device);
				get_info(device, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &partition_max_sub_devices);
				get_info(device, CL_DEVICE_PARTITION_PROPERTIES, &partition_properties);
				get_info(device, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &partition_affinity_domain);
				get_info(device, CL_DEVICE_PARTITION_TYPE, &partition_type);
				get_info(device, CL_DEVICE_REFERENCE_COUNT, &reference_count);
				get_info(device, CL_DEVICE_SVM_CAPABILITIES, &svm_capabilities);
				get_info(device, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, &preferred_platform_atomic_alignment);
				get_info(device, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, &preferred_global_atomic_alignment);
				get_info(device, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, &preferred_local_atomic_alignment);
			}
			template<typename T>
			void get_info(const cl::Device &device, cl_device_info name, T *param) const
			{
				const cl_int status = device.getInfo(name, param);
				CHECK_OPENCL_STATUS(status);
			}
			const std::string& get_features()
			{
				if (m_features.empty())
				{
					print_field("type", type);
					print_field("vendor_id", vendor_id);
					print_field("max_compute_units", max_compute_units);
					print_field("max_work_item_dimensions", max_work_item_dimensions);
					print_array("max_work_item_sizes", max_work_item_sizes);
					print_field("max_work_group_size", max_work_group_size);

					print_field("preferred_vector_width_char", preferred_vector_width_char);
					print_field("preferred_vector_width_short", preferred_vector_width_short);
					print_field("preferred_vector_width_int", preferred_vector_width_int);
					print_field("preferred_vector_width_long", preferred_vector_width_long);
					print_field("preferred_vector_width_half", preferred_vector_width_half);
					print_field("preferred_vector_width_float", preferred_vector_width_float);
					print_field("preferred_vector_width_double", preferred_vector_width_double);

					print_field("native_vector_width_char", native_vector_width_char);
					print_field("native_vector_width_short", native_vector_width_short);
					print_field("native_vector_width_int", native_vector_width_int);
					print_field("native_vector_width_long", native_vector_width_long);
					print_field("native_vector_width_half", native_vector_width_half);
					print_field("native_vector_width_float", native_vector_width_float);
					print_field("native_vector_width_double", native_vector_width_double);

					print_field("max_clock_frequency", max_clock_frequency);
					print_field("address_bits", address_bits);
					print_field("max_mem_alloc_size", max_mem_alloc_size);

					print_field("image_support", image_support);
					print_field("max_read_image_args", max_read_image_args);
					print_field("max_write_image_args", max_write_image_args);
					print_field("max_read_write_image_args", max_read_write_image_args);
					print_field("image2D_max_width", image2D_max_width);
					print_field("image2D_max_height", image2D_max_height);
					print_field("image3D_max_width", image3D_max_width);
					print_field("image3D_max_height", image3D_max_height);
					print_field("image3D_max_depth", image3D_max_depth);
					print_field("image_max_buffer_size", image_max_buffer_size);
					print_field("image_max_array_size", image_max_array_size);
					print_field("max_samplers", max_samplers);
					print_field("image_pitch_alignment", image_pitch_alignment);
					print_field("image_base_address_alignment", image_base_address_alignment);

					print_field("max_pipe_args", max_pipe_args);
					print_field("pipe_max_active_reservations", pipe_max_active_reservations);
					print_field("pipe_max_packet_size", pipe_max_packet_size);

					print_field("max_parameter_size", max_parameter_size);
					print_field("mem_base_add_align", mem_base_add_align);

					print_field("single_fp_config", single_fp_config);
					print_field("double_fp_config", double_fp_config);

					print_field("global_mem_cache_type", global_mem_cache_type);
					print_field("global_mem_cacheline_size", global_mem_cacheline_size);
					print_field("global_mem_cache_size", global_mem_cache_size);
					print_field("global_mem_size", global_mem_size);

					print_field("max_constant_buffer_size", max_constant_buffer_size);
					print_field("max_constant_args", max_constant_args);

					print_field("max_global_variable_size", max_global_variable_size);
					print_field("global_variable_preferred_total_size", global_variable_preferred_total_size);

					print_field("local_mem_type", local_mem_type);
					print_field("local_mem_size", local_mem_size);

					print_field("error_correction_support", error_correction_support);

					print_field("profiling_timer_resolution", profiling_timer_resolution);
					print_field("endian_little", endian_little);
					print_field("available", available);

					print_field("compiler_available", compiler_available);
					print_field("linker_available", linker_available);

					print_field("execution_capabilities", execution_capabilities);
					print_field("queue_on_host_properties", queue_on_host_properties);
					print_field("queue_on_device_properties", queue_on_device_properties);
					print_field("queue_on_device_max_size", queue_on_device_max_size);
					print_field("max_on_device_queues", max_on_device_queues);
					print_field("max_on_device_events", max_on_device_events);

					print_field("built_in_kernels", built_in_kernels);
					print_hex("platform", platform);
					print_field("name", name);
					print_field("vendor", vendor);
					print_field("driver_version", driver_version);
					print_field("profile", profile);
					print_field("version", version);
					print_field("opencl_c_version", opencl_c_version);
					print_field("extensions", extensions);

					print_field("printf_buffer_size", printf_buffer_size);
					print_field("preferred_interop_user_sync", preferred_interop_user_sync);
					print_hex("parent_device", parent_device);
					print_field("partition_max_sub_devices", partition_max_sub_devices);
					print_array("partition_properties", partition_properties);
					print_field("partition_affinity_domain", partition_affinity_domain);
					print_array("partition_type", partition_type);

					print_field("reference_count", reference_count);
					print_field("svm_capabilities", svm_capabilities);
					print_field("preferred_platform_atomic_alignment", preferred_platform_atomic_alignment);
					print_field("preferred_global_atomic_alignment", preferred_global_atomic_alignment);
					print_field("preferred_local_atomic_alignment", preferred_local_atomic_alignment);
				}
				return m_features;
			}
		private:
			std::string m_features;

			template<typename T>
			void print_field(const std::string &name, const T &x)
			{
				m_features += name + " : " + std::to_string(x) + '\n';
			}
			void print_field(const std::string &name, const std::string &x)
			{
				m_features += name + " : \"" + x + "\"\n";
			}

			template<typename T>
			void print_hex(const std::string &name, T x)
			{
				m_features += name + " : 0x";
				for (size_t i = 0; i < sizeof(T); i++)
					m_features += to_hex(reinterpret_cast<const uint8_t*>(&x)[i]);
				m_features += "\n";
			}
			template<typename T>
			void print_array(const std::string &name, const std::vector<T> &x)
			{
				m_features += name + " : [";
				for (size_t i = 0; i < x.size(); i++)
					m_features += ((i == 0) ? "" : ", ") + std::to_string(x[i]);
				m_features += "]\n";
			}
			template<typename T>
			void print_bits(const std::string &name, T x)
			{
				m_features += name + " : " + std::bitset<8 * sizeof(T)>(x).to_string() + '\n';
			}
	};

	std::vector<openclDeviceProp>& get_device_properties()
	{
		static std::vector<openclDeviceProp> properties = []()
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
		if (0 <= index and index < opencl_get_number_of_devices())
		{
			switch (dtype)
			{
				case DTYPE_FLOAT16:
				{
//					static const bool result = get_device_properties().at(index).extensions.find("cl_khr_fp16") != std::string::npos;
//					return result;
					return false; // TODO
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
		if (0 <= index and index < opencl_get_number_of_devices())
			return infos.at(index).data();
		else
			return nullptr;
	}
	const char* opencl_get_device_features(int index)
	{
		if (0 <= index and index < opencl_get_number_of_devices())
			return get_device_properties().at(index).get_features().data();
		else
			return nullptr;
	}

} /* namespace ml */

