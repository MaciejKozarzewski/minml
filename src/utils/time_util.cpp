/*
 * time_util.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/time_util.hpp>
#include <cmath>
#include <chrono>

double getTime()
{
	auto current_time = std::chrono::system_clock::now();
	auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());

	return duration_in_seconds.count();
}
std::string formatTime(double seconds, int precision)
{
	int s = static_cast<int>(seconds) % 60;
	int m = (static_cast<int>(seconds) % 3600) / 60;
	int h = static_cast<int>(seconds) / 3600;

	std::string result;
	if (h < 10)
		result += "0";
	result += std::to_string(h) + ":";
	if (m < 10)
		result += "0";
	result += std::to_string(m) + ":";
	if (s < 10)
		result += "0";
	result += std::to_string(s);

	if (precision > 0)
	{
		int decimal = static_cast<int>(pow(10, precision) * (seconds - static_cast<int>(seconds)) + 0.5);
		result += std::string(".") + std::to_string(decimal);
	}
	return result;
}

