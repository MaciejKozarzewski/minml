/*
 * DataType.cpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/DataType.hpp>
#include <minml/core/ml_exceptions.hpp>

#include <vector>
#include <cstring>

namespace
{
	uint32_t convertmantissa(uint32_t i) noexcept
	{
		uint32_t m = i << 13; // Zero pad mantissa bits
		uint32_t e = 0; // Zero exponent
		while (!(m & 0x00800000))
		{
			e -= 0x00800000;
			m <<= 1;
		}
		m &= ~0x00800000;
		e += 0x38800000;
		return m | e;
	}
	std::vector<uint32_t> init_mantissa_table()
	{
		std::vector<uint32_t> result(2048);
		result[0] = 0;
		uint32_t i = 1;
		for (; i <= 1023; i++)
			result[i] = convertmantissa(i);
		for (; i < 2048; i++)
			result[i] = 0x38000000 + ((i - 1024) << 13);
		return result;
	}
	std::vector<uint32_t> init_exponent_table()
	{
		std::vector<uint32_t> result(64);
		result[0] = 0;
		for (uint32_t i = 1; i <= 30; i++)
			result[i] = i << 23;
		result[31] = 0x47800000;
		result[32] = 0x80000000;
		for (uint32_t i = 33; i <= 62; i++)
			result[i] = 0x80000000 + ((i - 32) << 23);
		result[63] = 0xC7800000;
		return result;
	}
	std::vector<uint16_t> init_offset_table()
	{
		std::vector<uint16_t> result(64, 1024);
		result[0] = 0;
		result[32] = 0;
		return result;
	}
	std::vector<uint16_t> init_base_table()
	{
		std::vector<uint16_t> result(512);
		for (uint32_t i = 0; i < 256; i++)
		{
			int32_t e = i - 127;
			if (e < -24)
			{
				result[i | 0x000] = 0x0000;
				result[i | 0x100] = 0x8000;
			}
			else
			{
				if (e < -14)
				{
					result[i | 0x000] = (0x0400 >> (-e - 14));
					result[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
				}
				else
				{
					if (e <= 15)
					{
						result[i | 0x000] = ((e + 15) << 10);
						result[i | 0x100] = ((e + 15) << 10) | 0x8000;
					}
					else
					{
						if (e < 128)
						{
							result[i | 0x000] = 0x7C00;
							result[i | 0x100] = 0xFC00;
						}
						else
						{
							result[i | 0x000] = 0x7C00;
							result[i | 0x100] = 0xFC00;
						}
					}
				}
			}
		}
		return result;
	}
	std::vector<uint8_t> init_shift_table()
	{
		std::vector<uint8_t> result(512);
		for (uint32_t i = 0; i < 256; i++)
		{
			int32_t e = i - 127;
			if (e < -24)
			{
				result[i | 0x000] = 24;
				result[i | 0x100] = 24;
			}
			else
			{
				if (e < -14)
				{
					result[i | 0x000] = -e - 1;
					result[i | 0x100] = -e - 1;
				}
				else
				{
					if (e <= 15)
					{
						result[i | 0x000] = 13;
						result[i | 0x100] = 13;
					}
					else
					{
						if (e < 128)
						{
							result[i | 0x000] = 24;
							result[i | 0x100] = 24;
						}
						else
						{
							result[i | 0x000] = 13;
							result[i | 0x100] = 13;
						}
					}
				}
			}
		}
		return result;
	}
}

namespace ml
{

	uint16_t convert_fp32_to_fp16(float x)
	{
		static const std::vector<uint16_t> base_table = init_base_table();
		static const std::vector<uint8_t> shift_table = init_shift_table();

		uint32_t tmp = 0;
		std::memcpy(&tmp, &x, sizeof(tmp));
		return base_table[(tmp >> 23) & 0x1ff] + ((tmp & 0x007fffff) >> shift_table[(tmp >> 23) & 0x1ff]);
	}
	uint16_t convert_fp32_to_bf16(float x)
	{
		uint16_t tmp[2];
		std::memcpy(&tmp, &x, sizeof(float));
		return tmp[1];
	}
	float convert_fp16_to_fp32(uint16_t x)
	{
		static const std::vector<uint32_t> mantissa_table = init_mantissa_table();
		static const std::vector<uint16_t> offset_table = init_offset_table();
		static const std::vector<uint32_t> exponent_table = init_exponent_table();

		const uint32_t tmp = mantissa_table[offset_table[x >> 10] + (x & 0x03ff)] + exponent_table[x >> 10];
		float result;
		std::memcpy(&result, &tmp, sizeof(result));
		return result;
	}
	float convert_bf16_to_fp32(uint16_t x)
	{
		uint16_t tmp[2] = { 0u, x };
		float result;
		std::memcpy(&result, tmp, sizeof(float));
		return result;
	}

	size_t sizeOf(DataType t) noexcept
	{
		switch (t)
		{
			case DataType::BFLOAT16:
			case DataType::FLOAT16:
				return 2;
			case DataType::FLOAT32:
			case DataType::INT32:
				return 4;
			default:
				return 0;
		}
	}

	DataType typeFromString(const std::string &str)
	{
		if (str == "bf16" or str == "bfloat16" or str == "BFLOAT16")
			return DataType::BFLOAT16;
		if (str == "fp16" or str == "float16" or str == "FLOAT16")
			return DataType::FLOAT16;
		if (str == "fp32" or str == "float32" or str == "FLOAT32")
			return DataType::FLOAT32;
		if (str == "int32" or str == "INT32")
			return DataType::INT32;
		throw DataTypeNotSupported(METHOD_NAME, "unknown data type '" + str + "'");
	}
	std::string toString(DataType t)
	{
		switch (t)
		{
			case DataType::BFLOAT16:
				return std::string("BFLOAT16");
			case DataType::FLOAT16:
				return std::string("FLOAT16");
			case DataType::FLOAT32:
				return std::string("FLOAT32");
			case DataType::INT32:
				return std::string("INT32");
			default:
				return std::string("UNKNOWN");
		}
	}

	std::ostream& operator<<(std::ostream &stream, DataType t)
	{
		stream << toString(t);
		return stream;
	}
	std::string operator+(const std::string &lhs, DataType rhs)
	{
		return lhs + toString(rhs);
	}
	std::string operator+(DataType lhs, const std::string &rhs)
	{
		return toString(lhs) + rhs;
	}

	DataTypeNotSupported::DataTypeNotSupported(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, const char *comment) :
			DataTypeNotSupported(function, std::string(comment))
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, DataType dtype) :
			std::logic_error(std::string(function) + " : " + dtype + " is not supported")
	{
	}

	DataTypeMismatch::DataTypeMismatch(const char *function, const std::string &comment) :
			logic_error(std::string(function) + " : " + comment)
	{
	}
	DataTypeMismatch::DataTypeMismatch(const char *function, DataType d1, DataType d2) :
			std::logic_error(std::string(function) + " : expected type " + d1 + ", got " + d2)
	{
	}

} /* namespace ml */
