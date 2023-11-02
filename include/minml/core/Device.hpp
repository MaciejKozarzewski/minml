/*
 * Device.hpp
 *
 *  Created on: May 12, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_DEVICE_HPP_
#define MINML_CORE_DEVICE_HPP_

#include <string>
#include <stdexcept>

namespace ml /* forward declarations */
{
	class DeviceContext;
	enum class DataType;
}

namespace ml
{
	enum class DeviceType
	{
		CPU,
		CUDA,
		OPENCL
	};

	enum class CpuSimd
	{
		NONE,
		SSE,
		SSE2,
		SSE3,
		SSSE3,
		SSE41,
		SSE42,
		AVX,
		AVX2,
		AVX512F,
		AVX512VL_BW_DQ
	};

	class Device
	{
		private:
			DeviceType m_type;
			int m_index;
			Device(DeviceType type, int index);
		public:
			// device creation
			static Device cpu() noexcept;
			static Device cuda(int index);
			static Device opencl(int index);
			static Device fromString(const std::string &str);

			DeviceType type() const noexcept;
			int index() const noexcept;
			bool isCPU() const noexcept;
			bool isCUDA() const noexcept;
			bool isOPENCL() const noexcept;

			bool supportsType(DataType t) const noexcept;

			std::string toString() const;
			std::string info() const;

			/*
			 * \brief In MB.
			 */
			int memory() const;
			static CpuSimd cpuSimdLevel();
			static int numberOfCpuCores();
			static int numberOfCudaDevices();
			static int numberOfOpenCLDevices();
			static void setNumberOfThreads(int t);
			static std::string hardwareInfo();

			friend bool operator==(const Device &lhs, const Device &rhs);
			friend bool operator!=(const Device &lhs, const Device &rhs);
	};

	std::ostream& operator<<(std::ostream &stream, const Device &d);
	std::string operator+(const std::string &str, const Device &d);
	std::string operator+(const Device &d, const std::string &str);

	class DeviceMismatch: public std::logic_error
	{
		public:
			DeviceMismatch(const char *function);
			DeviceMismatch(const char *function, const std::string &comment);
	};
	class IllegalDevice: public std::invalid_argument
	{
		public:
			IllegalDevice(const char *function, Device d);
	};

} /* namespace ml */

#endif /* MINML_CORE_DEVICE_HPP_ */
