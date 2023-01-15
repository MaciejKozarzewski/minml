/*
 * ml_exceptions.h
 *
 *  Created on: May 7, 2020
 *      Author: maciek
 */

#ifndef MINML_CORE_ML_EXCEPTIONS_HPP_
#define MINML_CORE_ML_EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>
#include <cassert>

namespace ml
{
#ifdef __GNUC__
#  define METHOD_NAME __PRETTY_FUNCTION__
#else
#  define METHOD_NAME __FUNCTION__
#endif

//runtime errors
	class RuntimeError: public std::runtime_error
	{
		public:
			RuntimeError(const char *function);
			RuntimeError(const char *function, const std::string &comment);
	};

	//range errors
	class IndexOutOfBounds: public std::out_of_range
	{
		public:
			IndexOutOfBounds(const char *function, const std::string &index_name, int index_value, int range);
	};
	class OutOfRange: public std::out_of_range
	{
		public:
			OutOfRange(const char *function, int value, int range);
	};

	//not-supported errors
	class NotImplemented: public std::logic_error
	{
		public:
			NotImplemented(const char *function);
			NotImplemented(const char *function, const std::string &comment);
	};

	class LogicError: public std::logic_error
	{
		public:
			LogicError(const char *function);
			LogicError(const char *function, const std::string &comment);
	};

	class UninitializedObject: public std::logic_error
	{
		public:
			UninitializedObject(const char *function);
			UninitializedObject(const char *function, const std::string &comment);
	};

	//illegal argument
	class IllegalArgument: public std::invalid_argument
	{
		public:
			IllegalArgument(const char *function, const std::string &comment);
			IllegalArgument(const char *function, const char *arg_name, const std::string &comment, int arg_value);
			IllegalArgument(const char *function, const char *arg_name, const std::string &comment, const std::string &arg_value);
	};

} /* namespace ml */

#endif /* MINML_CORE_ML_EXCEPTIONS_HPP_ */
