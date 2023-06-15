/*
 * version.cpp
 *
 *  Created on: Jun 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/version.hpp>

namespace ml
{
	Version getVersion() noexcept
	{
		return Version { 1, 0, 0 };
	}

} /* namespace ml */

