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
#if defined(_WIN32)
			freopen("NUL", "w", stdout); // redirect stdout to the windows version of /dev/null
#elif defined(__linux__)
			FILE *tmp = stderr;
			stderr = tmpfile();
#endif

			std::vector<cl::Device> result;
			const cl_int status = p.getDevices(CL_DEVICE_TYPE_GPU, &result);
			CHECK_OPENCL_STATUS(status);

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
			static cl::Context context = []()
			{
				cl_int status;
				cl::Context result(get_list_of_devices(), nullptr, nullptr, nullptr, &status);
				CHECK_OPENCL_STATUS(status);
				return result;
			}();
			return context;
		}

		void waitForEvent(const cl::Event *event)
		{
			assert(event != nullptr);
			const cl_int status = event->wait();
			CHECK_OPENCL_STATUS(status);
		}
		cl::Program compileProgram(const std::string &name, const std::string &source, const std::string &options)
		{
			cl_int status;
			cl::Program result(opencl::get_cl_context(), source, false, &status);
			CHECK_OPENCL_STATUS(status);

			const std::string all_options = options + " -cl-mad-enable";
			status = result.build(get_list_of_devices(), all_options.c_str());
			if (status != CL_SUCCESS)
			{
				std::cout << "Compilation status = " << status << '\n';
				const auto info = result.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
				for (size_t i = 0; i < info.size(); i++)
				{
					std::cout << "Build log for program " << name << " on device OPENCL:" << i << " - " << opencl_get_device_info(i) << ":\n";
					std::cout << info[i].second << '\n';
				}
				CHECK_OPENCL_STATUS(status);
			}
			return result;
		}
		cl::Kernel getKernel(const cl::Program &program, const char *name)
		{
			cl_int status;
			cl::Kernel result(program, name, &status);
			CHECK_OPENCL_STATUS(status);
			return result;
		}
		void runKernel(mlContext_t context, const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local)
		{
			cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
			cl::Event *event = opencl::Context::getLastEvent(context);
			const cl_int status = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, event);
			CHECK_OPENCL_STATUS(status);
		}

		/*
		 * Context
		 */
		Context::Context(int device_index) :
				m_device_index(device_index)
		{
			cl_int status;
			m_command_queue = cl::CommandQueue(get_cl_context(), get_list_of_devices().at(device_index), 0, &status);
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
		cl::CommandQueue& Context::getCommandQueue(mlContext_t context)
		{
			if (context == nullptr)
			{
				static cl::CommandQueue default_queue = []()
				{
					cl_int status;
					cl::CommandQueue result(get_cl_context(), get_list_of_devices().at(0), 0, &status);
					CHECK_OPENCL_STATUS(status);
					return result;
				}();
				return default_queue;
			}
			else
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
			{
				static cl::Buffer buffer = []()
				{
					cl_int status;
					cl::Buffer result(get_cl_context(), CL_MEM_READ_WRITE, 1024, nullptr, &status); // @suppress("Ambiguous problem")
					CHECK_OPENCL_STATUS(status);
					return result;
				}();
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

