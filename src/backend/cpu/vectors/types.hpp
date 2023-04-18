/*
 * types.hpp
 *
 *  Created on: Mar 29, 2023
 *      Author: maciek
 */

#ifndef VECTORS_TYPES_HPP_
#define VECTORS_TYPES_HPP_

#include <cinttypes>

/*
 * Natively supported bfloat16
 */
struct float16
{
		uint16_t m_data;

		friend bool operator==(float16 lhs, float16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(float16 lhs, float16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};
/*
 * Emulated float16 with software conversion
 */
struct sw_float16
{
		uint16_t m_data;

		friend bool operator==(sw_float16 lhs, sw_float16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(sw_float16 lhs, sw_float16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};

/*
 * Natively supported bfloat16
 */
struct bfloat16
{
		uint16_t m_data;

		friend bool operator==(bfloat16 lhs, bfloat16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(bfloat16 lhs, bfloat16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};
/*
 * Software emulated bfloat16
 */
struct sw_bfloat16
{
		uint16_t m_data;

		friend bool operator==(sw_bfloat16 lhs, sw_bfloat16 rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		friend bool operator!=(sw_bfloat16 lhs, sw_bfloat16 rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
};

#endif /* VECTORS_TYPES_HPP_ */
