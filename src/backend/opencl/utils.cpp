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
			try
			{
				cl::Platform::get(&result);
			} catch (std::exception &e)
			{
			}
			return result;
		}
		std::vector<cl::Device> get_devices_for_platform(const cl::Platform &p)
		{
			// necessary to suppress unwanted printing from 'getDevices'
#if defined(_WIN32)
			freopen("NUL", "w", stdout); // redirect stdout to the windows version of /dev/null
#elif defined(__linux__)
			FILE *tmp = stderr;
			stderr = tmpfile();
#endif

			std::vector<cl::Device> result;
			try
			{
				p.getDevices(CL_DEVICE_TYPE_ALL, &result);
			} catch (std::exception &e)
			{
			}

#if defined(_WIN32)
			freopen("CON", "w", stdout); // redirect stdout back to the console
#elif defined(__linux__)
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
		cl::Context& get_cl_context()
		{
			static cl::Context context(get_list_of_devices());
			return context;
		}
		cl::Program compile_program(const std::string &name, const std::string &source, const std::string &options)
		{
			cl::Program result(opencl::get_cl_context(), source);

			try
			{
				const cl_int status = result.build(get_list_of_devices(), options.c_str());
				if (status != CL_SUCCESS)
				{
					std::cout << "Compilation status = " << status << '\n';
					throw std::runtime_error("");
				}
			} catch (std::exception &e)
			{
				const auto info = result.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
				for (size_t i = 0; i < info.size(); i++)
				{
					std::cout << "Build log for program " << name << " on device OPENCL:" << i << " - " << opencl_get_device_info(i) << ":\n";
					std::cout << info[i].second << '\n';
				}
			}
			return result;
		}

		/*
		 * Context
		 */
		Context::Context(int device_index) :
				m_command_queue(get_cl_context(), get_list_of_devices().at(device_index)),
				m_device_index(device_index)
		{
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
		cl::CommandQueue& Context::getCommandQueue(mlContext_t context)
		{
			if (context == nullptr)
			{
				static cl::CommandQueue default_queue(get_cl_context(), get_list_of_devices().at(0));
				return default_queue;
			}
			else
				return get(context)->m_command_queue;
		}
		cl::Buffer& Context::getWorkspace(mlContext_t context)
		{
			if (context == nullptr)
			{
				static cl::Buffer buffer(get_cl_context(), CL_MEM_READ_WRITE, 0);
				return buffer;
			}
			else
			{
				if (getWorkspaceSize(context) == 0)
					setWorkspaceSize(context, default_workspace_size);
				return get(context)->m_workspace;
			}
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
				get(context)->m_workspace = cl::Buffer(get_cl_context(), CL_MEM_READ_WRITE, bytes);
			}
		}

		bool has_fp16_math(mlContext_t context)
		{
			return false;
		}

	} /* namespace opencl */
} /* namespace ml */

