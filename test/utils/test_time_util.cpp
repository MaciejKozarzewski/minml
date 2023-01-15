/*
 * test_time_util.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */
#include <gtest/gtest.h>
#include <minml/utils/time_util.hpp>

namespace ml
{
	TEST(TestTimeUtil, format_time)
	{
		double time = 3600 + 60 + 1.234;

		std::string tmp0 = formatTime(time, 0);
		std::string tmp1 = formatTime(time, 1);
		std::string tmp2 = formatTime(time, 2);
		std::string tmp3 = formatTime(time, 3);
		EXPECT_EQ(tmp0, "01:01:01");
		EXPECT_EQ(tmp1, "01:01:01.2");
		EXPECT_EQ(tmp2, "01:01:01.23");
		EXPECT_EQ(tmp3, "01:01:01.234");

		EXPECT_EQ(formatTime(11 * 3600 + 12 * 60 + 50.4, 1), "11:12:50.4");

		EXPECT_EQ(formatTime(0), "00:00:00");
	}

} /* namespace ml */

