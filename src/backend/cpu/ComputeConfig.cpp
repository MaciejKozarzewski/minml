/*
 * ComputeConfig.cpp
 *
 *  Created on: Apr 17, 2023
 *      Author: Maciej Kozarzewski
 */

#include "ComputeConfig.hpp"
#include "utils.hpp"
#include "cpu_x86.hpp"

#include <memory>
#include <stdexcept>
#include <functional>
#include <cassert>

namespace
{
	ml::cpu::Type convert(ml::mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case ml::DTYPE_UNKNOWN:
				return ml::cpu::Type::NONE;
			case ml::DTYPE_FLOAT16:
				return ml::cpu::Type::FP16;
			case ml::DTYPE_FLOAT32:
				return ml::cpu::Type::FP32;
			case ml::DTYPE_INT32:
				return ml::cpu::Type::INT32;
		}
	}
	std::string to_string(ml::cpu::Type t)
	{
		switch (t)
		{
			default:
			case ml::cpu::Type::NONE:
				return "none";
			case ml::cpu::Type::FP16:
				return "fp16";
			case ml::cpu::Type::FP32:
				return "fp32";
			case ml::cpu::Type::INT8:
				return "int8";
			case ml::cpu::Type::INT32:
				return "int32";
		}
	}

}

namespace ml
{
	namespace cpu
	{

		ComputeConfig::ComputeConfig(Type dataType, Type computeType) noexcept :
				data_type(dataType),
				compute_type(computeType)
		{
		}
		ComputeConfig::ComputeConfig(mlDataType_t dataType, mlDataType_t computeType) :
				data_type(convert(dataType)),
				compute_type(convert(computeType))
		{
		}
		std::string ComputeConfig::toString() const
		{
			return "data type = " + to_string(data_type) + ", compute type = " + to_string(compute_type);
		}
		ComputeConfig ComputeConfig::getBest(mlDataType_t dataType)
		{
			switch (dataType)
			{
				default:
					throw std::runtime_error("Unsupported compute configuration");
				case DTYPE_FLOAT16:
					if (has_hardware_fp16_math())
						return ComputeConfig(Type::FP16, Type::FP16);
					else
					{
						if (has_hardware_fp16_conversion())
							return ComputeConfig(Type::FP16, Type::FP32);
						else
							throw std::runtime_error("Unsupported compute configuration");
					}
				case DTYPE_FLOAT32:
					return ComputeConfig(Type::FP32, Type::FP32);
			}
		}

	} /* namespace cpu */
} /* namespace ml */
