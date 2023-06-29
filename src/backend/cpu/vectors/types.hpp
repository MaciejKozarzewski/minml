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

#endif /* VECTORS_TYPES_HPP_ */
