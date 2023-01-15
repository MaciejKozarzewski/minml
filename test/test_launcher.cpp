/*
 * test_launcher.cpp
 *
 *  Created on: May 7, 2020
 *      Author: Maciej Kozarzewski
 */
#include <minml/core/Device.hpp>

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
	ml::Device::setNumberOfThreads(1);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

