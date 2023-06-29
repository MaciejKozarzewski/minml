/*
 * ComputeConfig.hpp
 *
 *  Created on: Apr 17, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_COMPUTECONFIG_HPP_
#define BACKEND_CPU_COMPUTECONFIG_HPP_

#include <minml/backend/backend_types.h>

#include "vectors/types.hpp"

#include <functional>
#include <string>

namespace ml
{
	namespace cpu
	{
		enum class Type
		{
			NONE,
			FP16,
			FP32,
			INT8,
			INT32,
		};

		struct ComputeConfig
		{
				Type data_type = Type::NONE;
				Type compute_type = Type::NONE;

				ComputeConfig() noexcept = default;
				ComputeConfig(Type dataType, Type computeType) noexcept;
				ComputeConfig(mlDataType_t dataType, mlDataType_t computeType);

				std::string toString() const;
				static ComputeConfig getBest(mlDataType_t dataType);
		};

		template<typename T>
		ml::cpu::Type get_type() noexcept
		{
			if (std::is_same<T, float16>::value)
				return ml::cpu::Type::FP16;
			if (std::is_same<T, float>::value)
				return ml::cpu::Type::FP32;
			if (std::is_same<T, int8_t>::value)
				return ml::cpu::Type::INT8;
			if (std::is_same<T, int32_t>::value)
				return ml::cpu::Type::INT32;
			return ml::cpu::Type::NONE;
		}

		template<typename ... Args>
		class FunctionTable
		{
				using Function_t = std::function<void(Args...)>;
				std::array<std::array<Function_t, 8>, 8> table;
			public:
				Function_t& get(ml::cpu::Type dtype, ml::cpu::Type ctype)
				{
					return table[static_cast<int>(dtype)][static_cast<int>(ctype)];
				}
		};

		template<typename ... Args>
		FunctionTable<Args...> createFunctionTable(void (*)(Args...))
		{
			return FunctionTable<Args...>();
		}

#define CREATE_EMPTY_KERNEL_TABLE(name) static auto name##_table = ml::cpu::createFunctionTable(name<float, float>)

#define CREATE_KERNEL_TABLE(name) static auto name##_table = ml::cpu::createFunctionTable(name<float, float>);	\
		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP32) = name<float16, float>;						\
		name##_table.get(ml::cpu::Type::FP32, ml::cpu::Type::FP32) = name<float, float>

//#define CREATE_KERNEL_TABLE(name) static auto name##_table = ml::cpu::createFunctionTable(name<float, float>);	\
//		name##_table.get(ml::cpu::Type::SW_BF16, ml::cpu::Type::FP32) = name<sw_bfloat16, float>;				\
//		name##_table.get(ml::cpu::Type::BF16, ml::cpu::Type::FP32) = name<bfloat16, float>;						\
//		name##_table.get(ml::cpu::Type::SW_FP16, ml::cpu::Type::FP32) = name<sw_float16, float>;				\
//		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP32) = name<float16, float>;						\
//		name##_table.get(ml::cpu::Type::FP32, ml::cpu::Type::FP32) = name<float, float>

		/*
#define CREATE_KERNEL_TABLE(name) static auto name##_table = ml::cpu::createFunctionTable(name<float, float>);	\
		name##_table.get(ml::cpu::Type::SW_BF16, ml::cpu::Type::FP32) = name<sw_bfloat16, float>;		\
		name##_table.get(ml::cpu::Type::BF16, ml::cpu::Type::FP32) = name<bfloat16, float>;				\
		name##_table.get(ml::cpu::Type::BF16, ml::cpu::Type::BF16) = name<bfloat16, bfloat16>;			\
		name##_table.get(ml::cpu::Type::SW_FP16, ml::cpu::Type::FP32) = name<sw_float16, float>;		\
		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP32) = name<float16, float>;				\
		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP16) = name<float16, float16>;			\
		name##_table.get(ml::cpu::Type::FP32, ml::cpu::Type::FP32) = name<float, float>
		*/

#define REGISTER_KERNEL(name, dtype, ctype) name##_table.get(ml::cpu::get_type<dtype>(), ml::cpu::get_type<ctype>()) = name<dtype, ctype>
#define DISABLE_KERNEL(name, dtype, ctype) name##_table.get(ml::cpu::get_type<dtype>(), ml::cpu::get_type<ctype>()) = nullptr

#define CALL_KERNEL(name, cfg) name##_table.get(cfg.data_type, cfg.compute_type)

	}
} /* namespace ml */

#endif /* BACKEND_CPU_COMPUTECONFIG_HPP_ */
