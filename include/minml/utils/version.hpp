/*
 * version.hpp
 *
 *  Created on: Jun 15, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_VERSION_HPP_
#define MINML_VERSION_HPP_

namespace ml
{
	struct Version
	{
			int major;
			int minor;
			int revision;
	};

	Version getVersion() noexcept;

} /* namespace ml */

#endif /* MINML_VERSION_HPP_ */
