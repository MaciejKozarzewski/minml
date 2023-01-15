/*
 * time_util.hpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_TIME_UTIL_HPP_
#define MINML_UTILS_TIME_UTIL_HPP_

#include <string>

double getTime();
std::string formatTime(double seconds, int precision = 0);

#endif /* MINML_UTILS_TIME_UTIL_HPP_ */
