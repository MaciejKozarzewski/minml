/*
 * test_string_util.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <minml/utils/string_util.hpp>

namespace ml
{
	TEST(TestStringUtil, starts_with)
	{
		std::string tmp = "testowy_string";
		EXPECT_TRUE(startsWith(tmp, ""));
		EXPECT_TRUE(startsWith(tmp, "test"));
		EXPECT_FALSE(startsWith(tmp, "string"));
		std::string tmp2 = "test";
		EXPECT_TRUE(startsWith(tmp, tmp2));
	}
	TEST(TestStringUtil, ends_with)
	{
		std::string tmp = "testowy_string";
		EXPECT_TRUE(endsWith(tmp, ""));
		EXPECT_FALSE(endsWith(tmp, "test"));
		EXPECT_TRUE(endsWith(tmp, "string"));
		std::string tmp2 = "test";
		EXPECT_FALSE(endsWith(tmp, tmp2));
	}
	TEST(TestStringUtil, trim)
	{
		std::string tmp0 = "";
		std::string tmp1 = "  spacja  ";
		std::string tmp2 = "spacja  ";
		std::string tmp3 = "  spacja";
		std::string tmp4 = "spa cja";

		EXPECT_EQ(trim(tmp0), "");
		EXPECT_EQ(trim(tmp1), "spacja");
		EXPECT_EQ(trim(tmp2), "spacja");
		EXPECT_EQ(trim(tmp3), "spacja");
		EXPECT_EQ(trim(tmp4), "spa cja");
	}
	TEST(TestStringUtil, split)
	{
		std::string tmp = "to jest testowy string";
		auto s = split(tmp, ' ');
		EXPECT_EQ(s.size(), 4ull);
		EXPECT_EQ(s[0], "to");
		EXPECT_EQ(s[1], "jest");
		EXPECT_EQ(s[2], "testowy");
		EXPECT_EQ(s[3], "string");

		tmp = "";
		auto s2 = split(tmp, ' ');
		EXPECT_EQ(s2.size(), 1ull);
		EXPECT_EQ(s2[0], "");

		tmp = "  ";
		auto s3 = split(tmp, ' ');
		EXPECT_EQ(s3.size(), 3ull);
		EXPECT_EQ(s3[0], "");
		EXPECT_EQ(s3[1], "");
		EXPECT_EQ(s3[2], "");
	}

	TEST(TestStringUtil, isNumber)
	{
		EXPECT_FALSE(isNumber(""));
		EXPECT_FALSE(isNumber("a"));
		EXPECT_FALSE(isNumber("+"));
		EXPECT_FALSE(isNumber("-"));

		EXPECT_TRUE(isNumber(".0"));
		EXPECT_TRUE(isNumber(".1"));
		EXPECT_TRUE(isNumber("1."));
		EXPECT_TRUE(isNumber("+.0"));
		EXPECT_TRUE(isNumber("+.1"));
		EXPECT_TRUE(isNumber("+1."));
		EXPECT_FALSE(isNumber("."));
		EXPECT_FALSE(isNumber("+."));
		EXPECT_FALSE(isNumber("-."));

		EXPECT_TRUE(isNumber("-1234"));
		EXPECT_TRUE(isNumber("+1234"));
		EXPECT_TRUE(isNumber("1234"));

		EXPECT_TRUE(isNumber("-12.34"));
		EXPECT_TRUE(isNumber("+12.34"));
		EXPECT_TRUE(isNumber("12.34"));

		EXPECT_TRUE(isNumber("-12.34e+1"));
		EXPECT_TRUE(isNumber("+12.34e-1"));
		EXPECT_TRUE(isNumber("12.34e1"));
	}

} /* namespace ml */

