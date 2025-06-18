/*
 * utils.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include "utils.hpp"
#include <minml/backend/opencl_backend.h>

#include <CL/opencl.hpp>

#include <memory>
#include <cassert>
#include <iostream>

namespace
{
	ml::opencl::Context* get(ml::mlContext_t context)
	{
		return reinterpret_cast<ml::opencl::Context*>(context);
	}
}

namespace ml
{
	namespace opencl
	{
		std::vector<cl::Platform> get_list_of_platforms()
		{
			std::vector<cl::Platform> result;
			const cl_int status = cl::Platform::get(&result);
			CHECK_OPENCL_STATUS(status);
			return result;
		}
		std::vector<cl::Device> get_devices_for_platform(const cl::Platform &p)
		{
			// necessary to suppress unwanted printing from 'getDevices'
#if defined(__linux__)
			FILE *tmp = stderr;
			stderr = tmpfile();
#endif

			std::vector<cl::Device> result;
			const cl_int status = p.getDevices(CL_DEVICE_TYPE_GPU, &result);
			CHECK_OPENCL_STATUS(status);

#if defined(__linux__)
			fclose(stderr);
			stderr = tmp;
#endif
			return result;
		}
		const std::vector<cl::Device>& get_list_of_devices()
		{
			static const std::vector<cl::Device> devices = []()
			{
				std::vector<cl::Device> result;
				const auto platforms = get_list_of_platforms();
				for (size_t i = 0; i < platforms.size(); i++)
				{
					const auto tmp = get_devices_for_platform(platforms[i]);
					result.insert(result.end(), tmp.begin(), tmp.end());
				}
				return result;
			}();
			return devices;
		}
		std::vector<cl::Context>& get_list_of_contexts()
		{
			static std::vector<cl::Context> contexts = []()
			{
				cl_int status;
				std::vector<cl::Context> result;
				for (size_t i = 0; i < get_list_of_devices().size(); i++)
				{
					cl::Context tmp(get_list_of_devices().at(i), nullptr, nullptr, nullptr, &status);
					CHECK_OPENCL_STATUS(status);
					result.push_back(tmp);
				}
				return result;
			}();
			return contexts;
		}

		void waitForEvent(const cl::Event *event)
		{
			assert(event != nullptr);
			const cl_int status = event->wait();
			CHECK_OPENCL_STATUS(status);
		}
		void runKernel(mlContext_t context, const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local)
		{
			assert(context != nullptr);
			cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
			cl::Event *event = opencl::Context::getLastEvent(context);
			const cl_int status = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, event);
			if (status != CL_SUCCESS)
			{
				std::string name;
				kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &name);
				throw ml::opencl::OpenCLRuntimeError(name, __LINE__, status);
			}
			CHECK_OPENCL_STATUS(status);
		}

		/*
		 * Context
		 */
		Context::Context(int device_index) :
				m_device_index(device_index)
		{
			cl_int status;
			m_command_queue = cl::CommandQueue(get_list_of_contexts().at(device_index), get_list_of_devices().at(device_index),
			CL_QUEUE_PROFILING_ENABLE, &status);
			CHECK_OPENCL_STATUS(status);

			status = m_command_queue.enqueueMarkerWithWaitList(nullptr, &m_last_event);
			CHECK_OPENCL_STATUS(status);
		}
		void Context::synchronizeWith(mlContext_t context)
		{
			if (context != nullptr)
				waitForEvent(getLastEvent(context));
		}
		bool Context::isReady(mlContext_t context)
		{
			if (context == nullptr)
				return false;
			else
			{
				cl_int tmp = -1;
				const cl_int status = get(context)->m_last_event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &tmp);
				CHECK_OPENCL_STATUS(status);
				return tmp == CL_COMPLETE;
			}
		}
		int Context::getDeviceIndex(mlContext_t context)
		{
			if (context == nullptr)
				return -1;
			else
				return get(context)->m_device_index;
		}
		int Context::isSynchronized(mlContext_t context)
		{
			if (context == nullptr)
				return true;
			else
				return get(context)->m_is_synchronized;
		}
		cl::CommandQueue& Context::getDefaultCommandQueue(int device_index)
		{
			static std::vector<cl::CommandQueue> command_queues = []()
			{
				cl_int status;
				std::vector<cl::CommandQueue> result;
				for (size_t i = 0; i < get_list_of_devices().size(); i++)
				{
					cl::CommandQueue tmp(get_list_of_contexts().at(i), get_list_of_devices().at(i), CL_QUEUE_PROFILING_ENABLE, &status);
					CHECK_OPENCL_STATUS(status);
					result.push_back(tmp);
				}
				return result;
			}();
			return command_queues.at(device_index);
		}
		cl::CommandQueue& Context::getCommandQueue(mlContext_t context)
		{
			if (context == nullptr)
				throw std::logic_error("Context::getCommandQueue() got null context");

			return get(context)->m_command_queue;
		}
		cl::Event* Context::getLastEvent(mlContext_t context)
		{
			if (context == nullptr)
				return nullptr;
			else
				return &(get(context)->m_last_event);
		}
		cl::Buffer& Context::getWorkspace(mlContext_t context)
		{
			if (context == nullptr)
				throw std::logic_error("Context::getWorkspace() got null context");

			if (getWorkspaceSize(context) == 0)
				setWorkspaceSize(context, default_workspace_size);
			return get(context)->m_workspace;
		}
		size_t Context::getWorkspaceSize(mlContext_t context)
		{
			if (context == nullptr)
				return 0;
			else
				return get(context)->m_workspace_size;
		}
		void Context::setWorkspaceSize(mlContext_t context, size_t bytes)
		{
			if (context != nullptr and bytes > getWorkspaceSize(context))
			{
				get(context)->m_workspace_size = bytes;
				get(context)->m_workspace = cl::Buffer(get_list_of_contexts().at(getDeviceIndex(context)), CL_MEM_READ_WRITE, bytes);
			}
		}

		/*
		 * ProgramCache
		 */
		ProgramCache::ProgramCache(const std::string &name, const std::string &source, const std::string &options)
		{
			cl_int status;
			for (size_t i = 0; i < get_list_of_contexts().size(); i++)
			{
				cl::Program program(get_list_of_contexts().at(i), source, false, &status);
				CHECK_OPENCL_STATUS(status);

				const std::string all_options = options + " -cl-mad-enable";
				status = program.build(get_list_of_devices().at(i), all_options.c_str());
				if (status != CL_SUCCESS)
				{
					std::cout << "Compilation status = " << status << '\n';
					const auto info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
					for (size_t i = 0; i < info.size(); i++)
					{
						std::cout << "Build log for program " << name << " on device OPENCL:" << i << " - " << opencl_get_device_info(i) << ":\n";
						std::cout << info[i].second << '\n';
					}
					CHECK_OPENCL_STATUS(status);
				}
				m_programs.push_back(program);
			}
		}
		cl::Kernel ProgramCache::getKernel(mlContext_t context, const char *name) const
		{
			cl_int status;
			cl::Kernel result(m_programs.at(Context::getDeviceIndex(context)), name, &status);
			CHECK_OPENCL_STATUS(status);
			return result;
		}

		/*
		 * MemoryObject
		 */
		MemoryObject::MemoryObject(int device_index, size_t count) :
				m_size(count),
				m_device_index(device_index)
		{
			cl_int status;
			m_buffer = cl::Buffer(ml::opencl::get_list_of_contexts().at(device_index), CL_MEM_READ_WRITE, count, nullptr, &status); // @suppress("Ambiguous problem")
			CHECK_OPENCL_STATUS(status);
		}
		MemoryObject::MemoryObject(MemoryObject &other, size_t offset, size_t size) :
				m_buffer(other.m_buffer),
				m_offset(other.m_offset + offset),
				m_size(size),
				m_device_index(other.m_device_index)
		{
			assert(m_offset + m_size <= other.m_size);
		}
		const cl::Buffer& MemoryObject::buffer() const
		{
			if (m_offset == 0)
				return m_buffer;
			else
			{
				_cl_buffer_region tmp;
				tmp.origin = m_offset;
				tmp.size = m_size;

				cl_int status;
				m_view = m_buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmp, &status);
				CHECK_OPENCL_STATUS(status);
				return m_view;
			}
		}
		cl::Buffer& MemoryObject::buffer()
		{
			if (m_offset == 0)
				return m_buffer;
			else
			{
				_cl_buffer_region tmp;
				tmp.origin = m_offset;
				tmp.size = m_size;

				cl_int status;
				m_view = m_buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmp, &status);
				CHECK_OPENCL_STATUS(status);
				return m_view;
			}
		}

		bool has_fp16_math(mlContext_t context)
		{
			return false;
		}

	} /* namespace opencl */
} /* namespace ml */

