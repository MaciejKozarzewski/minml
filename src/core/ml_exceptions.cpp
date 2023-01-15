/*
 * ml_exceptions.cpp
 *
 *  Created on: May 8, 2020
 *      Author: maciek
 */

#include <minml/core/ml_exceptions.hpp>
#include <minml/core/DataType.hpp>
#include <iostream>

namespace ml
{
	//runtime errors
	RuntimeError::RuntimeError(const char *function) :
			runtime_error(function)
	{
	}
	RuntimeError::RuntimeError(const char *function, const std::string &comment) :
			runtime_error(std::string(function) + comment)
	{
	}

	//range errors
	IndexOutOfBounds::IndexOutOfBounds(const char *function, const std::string &index_name, int index_value, int range) :
			out_of_range(std::string(function) + " : '" + index_name + "' = " + std::to_string(index_value) + " out of range [0, " + std::to_string(range) + ")")
	{
	}
	OutOfRange::OutOfRange(const char *function, int value, int range) :
			out_of_range(std::string(function) + " : " + std::to_string(value) + " out of range [0, " + std::to_string(range) + ")")
	{
	}

	//not-supported errors
	NotImplemented::NotImplemented(const char *function) :
			logic_error(function)
	{
	}
	NotImplemented::NotImplemented(const char *function, const std::string &comment) :
			logic_error(std::string(function) + " : " + comment)
	{
	}

	LogicError::LogicError(const char *function) :
			std::logic_error(function)
	{
	}
	LogicError::LogicError(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

	UninitializedObject::UninitializedObject(const char *function) :
			std::logic_error(function)
	{
	}
	UninitializedObject::UninitializedObject(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

	//illegal argument
	IllegalArgument::IllegalArgument(const char *function, const std::string &comment) :
			invalid_argument(std::string(function) + " : " + comment)
	{
	}
	IllegalArgument::IllegalArgument(const char *function, const char *arg_name, const std::string &comment, int arg_value) :
			IllegalArgument(function, arg_name, comment, std::to_string(arg_value))
	{
	}
	IllegalArgument::IllegalArgument(const char *function, const char *arg_name, const std::string &comment, const std::string &arg_value) :
			invalid_argument(std::string(function) + " : '" + arg_name + "' " + comment + ", got " + arg_value)
	{
	}

}
/* namespace ml */

