/*
 * string_util.hpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_STRING_UTIL_HPP_
#define MINML_UTILS_STRING_UTIL_HPP_

#include <string>
#include <vector>

bool equals(const char *str1, const char *str2);
int occurence(const std::string &str, char c);

std::string toLowerCase(std::string s);

bool startsWith(const std::string &str, const std::string &seek);
bool endsWith(const std::string &str, const std::string &seek);
std::string trim(const std::string &str);

std::vector<std::string> split(const std::string &str, char delimiter);
bool isNumber(const std::string &str) noexcept;

void print_string(const std::string &str);
void println(const std::string &str);

#endif /* MINML_UTILS_STRING_UTIL_HPP_ */
