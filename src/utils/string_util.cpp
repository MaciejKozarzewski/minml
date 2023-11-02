/*
 * string_util.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/string_util.hpp>

#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

bool equals(const char *str1, const char *str2)
{
	return strcmp(str1, str2) == 0;
}
int occurence(const std::string &str, char c)
{
	int result = 0;
	for (size_t i = 0; i < str.length(); i++)
		result += str[i] == c;
	return result;
}

std::string toLowerCase(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
	{	return std::tolower(c);});
	return s;
}

bool startsWith(const std::string &str, const std::string &seek)
{
	if (seek.length() > str.length()) //string cannot start with substring longer than itself
		return false;
	for (size_t i = 0; i < seek.length(); i++)
		if (str[i] != seek[i])
			return false;
	return true;
}
bool endsWith(const std::string &str, const std::string &seek)
{
	if (seek.length() > str.length()) //string cannot start with substring longer than itself
		return false;
	size_t it0 = str.length() - seek.length();
	for (size_t it1 = 0; it1 < seek.length(); it0++, it1++)
		if (str[it0] != seek[it1])
			return false;
	return true;
}
std::string trim(const std::string &str)
{
	size_t start = 0, stop = str.length();
	for (; start < str.length(); start++)
		if (str[start] != ' ')
			break;
	for (; stop > 0; stop--)
		if (str[stop - 1] != ' ')
			break;
	return str.substr(start, stop - start);
}
std::vector<std::string> split(const std::string &str, char delimiter)
{
	std::vector<std::string> result(occurence(str, delimiter) + 1);
	size_t start = 0, count = 0;
	for (size_t stop = 0; stop <= str.length(); stop++)
		if (str[stop] == delimiter || stop == str.length())
		{
			result[count] = str.substr(start, stop - start);
			start = stop + 1;
			count++;
		}
	return result;
}
bool isNumber(const std::string &s) noexcept
{
	char *end = nullptr;
	double val = strtod(s.c_str(), &end);
	return end != s.c_str() && *end == '\0' && val != HUGE_VAL;
}

void print_string(const std::string &str)
{
	std::cout << str.data();
}
void println(const std::string &str)
{
	std::cout << str.data() << std::endl;
}

