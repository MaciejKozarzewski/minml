/*
 * cuda_properties.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>

#include "utils.hpp"

#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <cassert>
#include <iostream>

namespace
{
	int get_number_of_devices()
	{
		static const int number = []()
		{
			int result = 0;
			cudaError_t status = cudaGetDeviceCount(&result);
			if (status != cudaSuccess)
				return 0;
			return result;
		}();
		return number;
	}
	const std::vector<cudaDeviceProp>& get_device_properties()
	{
		static const std::vector<cudaDeviceProp> properties = []()
		{
			std::vector<cudaDeviceProp> result;
			const int count = get_number_of_devices();
			for (int i = 0; i < count; i++)
			{
				cudaDeviceProp prop;
				cudaError_t status = cudaGetDeviceProperties(&prop, i);
				assert(status == cudaSuccess);
				result.push_back(prop);
			}
			return result;
		}();
		return properties;
	}
	std::vector<std::string> get_device_infos()
	{
		std::vector<std::string> result;
		const int count = get_number_of_devices();
		for (int i = 0; i < count; i++)
		{
			cudaDeviceProp prop;
			cudaError_t status = cudaGetDeviceProperties(&prop, i);
			assert(status == cudaSuccess);

			std::string tmp = std::string(prop.name) + " : " + std::to_string(prop.multiProcessorCount) + " x ";
			tmp += "SM " + std::to_string(prop.major) + "." + std::to_string(prop.minor);
			tmp += " with " + std::to_string(prop.totalGlobalMem >> 20) + "MB of memory";
			result.push_back(tmp);
		}
		return result;
	}
}

namespace ml
{
	int cuda_get_number_of_devices()
	{
		static const int result = get_number_of_devices();
		return result;
	}
	bool cuda_supports_type(int index, mlDataType_t dtype)
	{
		if (index >= 0 and index < cuda_get_number_of_devices())
		{
			switch (dtype)
			{
				case DTYPE_BFLOAT16:
					return cuda::get_compute_capability(index) >= 53;
				case DTYPE_FLOAT16:
					return cuda::get_compute_capability(index) >= 53;
				case DTYPE_FLOAT32:
					return true;
			}
		}
		return false;
	}
	const char* cuda_get_device_info(int index)
	{
		static const std::vector<std::string> infos = get_device_infos();
		if (index >= 0 and index < cuda_get_number_of_devices())
			return infos.at(index).data();
		else
			return nullptr;
	}

	namespace cuda
	{
		// from utils.hpp
		int get_compute_capability(int device_index)
		{
			return (get_device_properties()[device_index].major * 10 + get_device_properties()[device_index].minor);
		}

		bool has_fp16_math(mlContext_t context)
		{
			const int sm_ver = cuda::get_compute_capability(cuda::Context::getDeviceIndex(context));
			return sm_ver == 53 or sm_ver == 60 or sm_ver >= 62;
		}

		bool has_bf16_math(mlContext_t context)
		{
			const int sm_ver = cuda::get_compute_capability(cuda::Context::getDeviceIndex(context));
			return sm_ver >= 80;
		}
	}
} /* namespace ml */

