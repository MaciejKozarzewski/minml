/*
 * ZipWrapper.hpp
 *
 *  Created on: Mar 7, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_ZIPWRAPPER_HPP_
#define MINML_UTILS_ZIPWRAPPER_HPP_

#include <vector>
#include <cstddef>

class ZipWrapper
{
	public:
		static const size_t CHUNK = 262144;
		static std::vector<char> compress(const std::vector<char> &data, int level = -1);
		static std::vector<char> uncompress(const std::vector<char> &data);
};

#endif /* MINML_UTILS_ZIPWRAPPER_HPP_ */
