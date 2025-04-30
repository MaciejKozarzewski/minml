/*
 * hardware.cpp
 *
 *  Created on: May 12, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Device.hpp>
#include <minml/core/DataType.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/string_util.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>
#include <minml/backend/opencl_backend.h>

#include <cstring>
#include <vector>

namespace ml
{

	Device::Device(DeviceType type, int index) noexcept :
			m_type(type),
			m_index(index)
	{
	}

//device creation
	Device Device::cpu() noexcept
	{
		return Device(DeviceType::CPU, 0);
	}
	Device Device::cuda(int index)
	{
		if (index < 0 or index >= Device::numberOfCudaDevices())
			throw IllegalDevice(METHOD_NAME, { DeviceType::CUDA, index });
		return Device(DeviceType::CUDA, index);
	}
	Device Device::opencl(int index)
	{
		if (index < 0 or index >= Device::numberOfOpenCLDevices())
			throw IllegalDevice(METHOD_NAME, { DeviceType::OPENCL, index });
		return Device(DeviceType::OPENCL, index);
	}
	Device Device::fromString(const std::string &str)
	{
		const std::string tmp = toLowerCase(str);
		if (tmp == "cpu")
			return Device::cpu();
		if (startsWith(tmp, "cuda:"))
			return Device::cuda(std::atoi(str.data() + 5));
		if (startsWith(tmp, "opencl:"))
			return Device::opencl(std::atoi(str.data() + 7));
		throw LogicError(METHOD_NAME, "Illegal device '" + str + "'");
	}

	int Device::index() const noexcept
	{
		return m_index;
	}
	DeviceType Device::type() const noexcept
	{
		return m_type;
	}
	bool Device::isCPU() const noexcept
	{
		return type() == DeviceType::CPU;
	}
	bool Device::isCUDA() const noexcept
	{
		return type() == DeviceType::CUDA;
	}
	bool Device::isOPENCL() const noexcept
	{
		return type() == DeviceType::OPENCL;
	}

//flags
	bool Device::supportsType(DataType t) const noexcept
	{
		switch (type())
		{
			case DeviceType::CPU:
				return cpu_supports_type(static_cast<mlDataType_t>(t));
			case DeviceType::CUDA:
				return cuda_supports_type(index(), static_cast<mlDataType_t>(t));
			case DeviceType::OPENCL:
				return opencl_supports_type(index(), static_cast<mlDataType_t>(t));
			default:
				return false;
		}
	}

	int Device::memory() const
	{
		switch (type())
		{
			case DeviceType::CPU:
				return cpu_get_memory();
			case DeviceType::CUDA:
				return cuda_get_memory(index());
			case DeviceType::OPENCL:
				return opencl_get_memory(index());
			default:
				return 0;
		}
	}
	CpuSimd Device::cpuSimdLevel()
	{
		static const CpuSimd result = static_cast<CpuSimd>(cpu_get_simd_level());
		return result;
	}
	int Device::numberOfCpuCores()
	{
		static const int result = cpu_get_number_of_cores();
		return result;
	}
	int Device::numberOfCudaDevices()
	{
		static const int result = cuda_get_number_of_devices();
		return result;
	}
	int Device::numberOfOpenCLDevices()
	{
		static const int result = opencl_get_number_of_devices();
		return result;
	}
	void Device::setNumberOfThreads(int t)
	{
		cpu_set_number_of_threads(t);
	}
	void Device::flushDenormalsToZero(bool b)
	{
		cpu_flush_denormals_to_zero(b);
	}
	std::string Device::hardwareInfo()
	{
		std::string result = Device::cpu().toString() + " : " + Device::cpu().info() + '\n';
		for (int i = 0; i < Device::numberOfCudaDevices(); i++)
			result += Device::cuda(i).toString() + " : " + Device::cuda(i).info() + '\n';
		for (int i = 0; i < Device::numberOfOpenCLDevices(); i++)
			result += Device::opencl(i).toString() + " : " + Device::opencl(i).info() + '\n';
		return result;
	}

	std::string Device::toString() const
	{
		switch (m_type)
		{
			case DeviceType::CPU:
				return "CPU";
			case DeviceType::CUDA:
				return "CUDA:" + std::to_string(index());
			case DeviceType::OPENCL:
				return "OPENCL:" + std::to_string(index());
		}
		return "Illegal device";
	}
	std::string Device::info() const
	{
		switch (type())
		{
			case DeviceType::CPU:
				return cpu_get_device_info();
			case DeviceType::CUDA:
				return cuda_get_device_info(index());
			case DeviceType::OPENCL:
				return opencl_get_device_info(index());
			default:
				return std::string();
		}
	}

	bool operator==(const Device &lhs, const Device &rhs) noexcept
	{
		return lhs.type() == rhs.type() and lhs.index() == rhs.index();
	}
	bool operator!=(const Device &lhs, const Device &rhs) noexcept
	{
		return not (lhs == rhs);
	}

	std::ostream& operator<<(std::ostream &stream, const Device &d)
	{
		stream << d.toString();
		return stream;
	}
	std::string operator+(const std::string &str, const Device &d)
	{
		return str + d.toString();
	}
	std::string operator+(const Device &d, const std::string &str)
	{
		return d.toString() + str;
	}

	DeviceMismatch::DeviceMismatch(const char *function) :
			std::logic_error(function)
	{
	}
	DeviceMismatch::DeviceMismatch(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}
	IllegalDevice::IllegalDevice(const char *function, Device d) :
			std::invalid_argument(std::string(function) + " : " + d.toString())
	{
	}

} /* namespace ml */
