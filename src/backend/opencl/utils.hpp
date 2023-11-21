/*
 * utils.hpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_OPENCL_UTILS_HPP_
#define BACKEND_OPENCL_UTILS_HPP_

#include <minml/backend/opencl_backend.h>

#include <CL/opencl.hpp>

#include <memory>
#include <algorithm>
#include <string>
#include <cassert>

namespace ml
{
	namespace opencl
	{
		std::vector<cl::Platform> get_list_of_platforms();
		std::vector<cl::Device> get_devices_for_platform(const cl::Platform &p);
		const std::vector<cl::Device>& get_list_of_devices();
		cl::Context& get_cl_context();

		void waitForEvent(const cl::Event *event);
		cl::Program compileProgram(const std::string &name, const std::string &source, const std::string &options);
		cl::Kernel getKernel(const cl::Program &program, const char *name);
		void runKernel(mlContext_t context, const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local = cl::NullRange);

		class Context
		{
				static constexpr size_t default_workspace_size = 8 * 1024 * 1024; // 8MB

				size_t m_workspace_size = 0;
				cl::Buffer m_workspace;
				cl::CommandQueue m_command_queue;
				cl::Event m_last_event;
				int m_device_index = 0;
				bool m_is_synchronized = false;
			public:
				Context(int device_index);
				static void synchronizeWith(mlContext_t context);
				static bool isReady(mlContext_t context);
				static int getDeviceIndex(mlContext_t context);
				static int isSynchronized(mlContext_t context);
				static cl::CommandQueue& getCommandQueue(mlContext_t context);
				static cl::Event* getLastEvent(mlContext_t context);
				static cl::Buffer& getWorkspace(mlContext_t context);
				static size_t getWorkspaceSize(mlContext_t context);
				static void setWorkspaceSize(mlContext_t context, size_t bytes);
		};

		template<unsigned int maxBlocks>
		unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
		{
			return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
		}
		bool has_fp16_math(mlContext_t context);

		template<int MaxDim0 = std::numeric_limits<int>::max()>
		cl::NDRange get_nd_range(int dim0) noexcept
		{
			assert(dim0 >= 0);
			return cl::NDRange(std::min(dim0, MaxDim0));
		}
		template<int MaxDim0 = std::numeric_limits<int>::max(), int MaxDim1 = std::numeric_limits<int>::max()>
		cl::NDRange get_nd_range(int dim0, int dim1) noexcept
		{
			assert(dim0 >= 0);
			assert(dim1 >= 0);
			return cl::NDRange(std::min(dim0, MaxDim0), std::min(dim1, MaxDim1));
		}
		template<int MaxDim0 = std::numeric_limits<int>::max(), int MaxDim1 = std::numeric_limits<int>::max(), int MaxDim2 =
				std::numeric_limits<int>::max()>
		cl::NDRange get_nd_range(int dim0, int dim1, int dim2) noexcept
		{
			assert(dim0 >= 0);
			assert(dim1 >= 0);
			assert(dim2 >= 0);
			return cl::NDRange(std::min(dim0, MaxDim0), std::min(dim1, MaxDim1), std::min(dim2, MaxDim2));
		}

		[[maybe_unused]] static cl::Buffer& getBuffer(void *buf) noexcept
		{
			assert(buf != nullptr);
			return *reinterpret_cast<cl::Buffer*>(buf);
		}
		[[maybe_unused]] static const cl::Buffer& getBuffer(const void *buf) noexcept
		{
			assert(buf != nullptr);
			return *reinterpret_cast<const cl::Buffer*>(buf);
		}

		class OpenCLRuntimeError: public std::runtime_error
		{
			public:
				OpenCLRuntimeError(const std::string &function, int line, int error_code) :
						std::runtime_error(
								"OpenCLRuntimeError occurred in function '" + function + "' at line " + std::to_string(line) + " got error code " + std::to_string(error_code))
				{
				}
		};

#define CHECK_OPENCL_STATUS(status) if (status != CL_SUCCESS) throw ml::opencl::OpenCLRuntimeError(__FUNCTION__, __LINE__, status);

	} /* namespace opencl */
} /* namespace ml */

#endif /* BACKEND_OPENCL_UTILS_HPP_ */
