/*
 * fp16.cpp
 *
 *  Created on: Aug 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include "fp16.hpp"

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
}

namespace ml
{
	namespace cpu
	{
		namespace internal
		{
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

		} /* namespace internal */
	} /* namespace cpu */
} /* namespace ml */

