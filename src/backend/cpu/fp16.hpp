/*
 * fp16.hpp
 *
 *  Created on: Aug 10, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_FP16_HPP_
#define BACKEND_CPU_FP16_HPP_

#include <vector>
#include <cinttypes>

namespace ml
{
	namespace cpu
	{
		namespace internal
		{
			std::vector<uint32_t> init_mantissa_table();
			std::vector<uint32_t> init_exponent_table();
			std::vector<uint16_t> init_offset_table();
			std::vector<uint16_t> init_base_table();
			std::vector<uint8_t> init_shift_table();

			template<typename T, typename U>
			T bitwise_cast(U x) noexcept
			{
				static_assert(sizeof(T) == sizeof(U), "Cannot cast types of different sizes");
				union
				{
						U u;
						T t;
				} tmp;
				tmp.u = x;
				return tmp.t;
			}
		} /* namespace internal */

		static inline uint16_t convert_fp32_to_fp16(float x)
		{
			static const std::vector<uint16_t> base_table = internal::init_base_table();
			static const std::vector<uint8_t> shift_table = internal::init_shift_table();

			const uint32_t tmp = internal::bitwise_cast<uint32_t>(x);
			return base_table[(tmp >> 23) & 0x1ff] + ((tmp & 0x007fffff) >> shift_table[(tmp >> 23) & 0x1ff]);
		}
		static inline float convert_fp16_to_fp32(uint16_t x)
		{
			static const std::vector<uint32_t> mantissa_table = internal::init_mantissa_table();
			static const std::vector<uint16_t> offset_table = internal::init_offset_table();
			static const std::vector<uint32_t> exponent_table = internal::init_exponent_table();

			const uint32_t tmp = mantissa_table[offset_table[x >> 10] + (x & 0x03ff)] + exponent_table[x >> 10];
			return internal::bitwise_cast<float>(tmp);
		}
	} /* namespace cpu */
} /* namespace ml */

#endif /* BACKEND_CPU_FP16_HPP_ */
