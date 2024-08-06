/*
 * DataType.h
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_DATATYPE_HPP_
#define MINML_CORE_DATATYPE_HPP_

#include <string>
#include <stdexcept>

namespace ml
{
	enum class DataType
	{
		UNKNOWN,
		FLOAT16,
		FLOAT32,
		FLOAT64,
		INT32
	};

	uint16_t convert_fp32_to_fp16(float x);
	float convert_fp16_to_fp32(uint16_t x);

	size_t sizeOf(DataType t) noexcept;

	DataType typeFromString(const std::string &str);
	std::string toString(DataType t);

	std::ostream& operator<<(std::ostream &stream, DataType dtype);
	std::string operator+(const std::string &lhs, DataType rhs);
	std::string operator+(DataType lhs, const std::string &rhs);

	class DataTypeNotSupported: public std::logic_error
	{
		public:
			DataTypeNotSupported(const char *function, const std::string &comment);
			DataTypeNotSupported(const char *function, const char *comment);
			DataTypeNotSupported(const char *function, DataType dtype);
	};

	class DataTypeMismatch: public std::logic_error
	{
		public:
			DataTypeMismatch(const char *function, const std::string &comment);
			DataTypeMismatch(const char *function, DataType d1, DataType d2);
	};

} /* namespace ml */

#endif /* MINML_CORE_DATATYPE_HPP_ */
