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
		cl::Program compile_program(const std::string &name, const std::string &source, const std::string &options);

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

		[[maybe_unused]] static cl::Buffer& get_buffer(void *buf) noexcept
		{
			assert(buf != nullptr);
			return *reinterpret_cast<cl::Buffer*>(buf);
		}
		[[maybe_unused]] static const cl::Buffer& get_buffer(const void *buf) noexcept
		{
			assert(buf != nullptr);
			return *reinterpret_cast<const cl::Buffer*>(buf);
		}

		class Context
		{
				static constexpr size_t default_workspace_size = 8 * 1024 * 1024; // 8MB

				cl::Buffer m_workspace;
				cl::CommandQueue m_command_queue;
				int m_device_index = 0;
				bool m_is_synchronized = false;
			public:
				Context(int device_index);
				static int getDeviceIndex(mlContext_t context);
				static int isSynchronized(mlContext_t context);
				static cl::CommandQueue& getCommandQueue(mlContext_t context);
				static cl::Buffer& getWorkspace(mlContext_t context);
				static size_t getWorkspaceSize(mlContext_t context);
				static void setWorkspaceSize(mlContext_t context, size_t bytes);
		};

		template<unsigned int maxBlocks>
		unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
		{
			return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
		}

	} /* namespace opencl */
} /* namespace ml */

#endif /* BACKEND_OPENCL_UTILS_HPP_ */
