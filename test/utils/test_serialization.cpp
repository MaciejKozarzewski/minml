/*
 * test_SerializedObject.cpp
 *
 *  Created on: May 6, 2020
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <minml/utils/serialization.hpp>

namespace ml
{
//	TEST(TestSerialization, init)
//	{
//		SerializedObject so1;
//		EXPECT_EQ(so1.size(), 0ull);
//		EXPECT_EQ(so1.capacity(), 0ull);
//		EXPECT_EQ(so1.getPosition(), 0ull);
//
//		SerializedObject so2(100);
//		EXPECT_EQ(so2.size(), 0ull);
//		EXPECT_EQ(so2.capacity(), 100ull);
//		EXPECT_EQ(so2.getPosition(), 0ull);
//	}
//	TEST(TestSerialization, save_load)
//	{
//		SerializedObject so;
//
//		so.save<char>(110);
//		EXPECT_EQ(so.getPosition(), 1ull);
//		so.save<short>(13210);
//		EXPECT_EQ(so.getPosition(), 3ull);
//		so.save<int>(2341230);
//		EXPECT_EQ(so.getPosition(), 7ull);
//		so.save<int64_t>(11234123412310ll);
//		EXPECT_EQ(so.getPosition(), 15ull);
//		so.save<float>(1.34123f);
//		EXPECT_EQ(so.getPosition(), 19ull);
//		so.save<double>(1.1234123412);
//		EXPECT_EQ(so.getPosition(), 27ull);
//
//		so.setPosition(0);
//		EXPECT_EQ(so.load<char>(), 110);
//		EXPECT_EQ(so.load<short>(), 13210);
//		EXPECT_EQ(so.load<int>(), 2341230);
//		EXPECT_EQ(so.load<int64_t>(), 11234123412310ll);
//		EXPECT_EQ(so.load<float>(), 1.34123f);
//		EXPECT_EQ(so.load<double>(), 1.1234123412);
//	}
//	TEST(TestSerialization, save_load_array)
//	{
//		SerializedObject so(1000);
//
//		int tmp[123];
//		for (int i = 0; i < 123; i++)
//			tmp[i] = i;
//		so.saveArray<int>(tmp, 123);
//		EXPECT_EQ(so.getPosition(), static_cast<size_t>(sizeof(int) * 123));
//
//		so.setPosition(0);
//		int loaded[123];
//		so.loadArray<int>(loaded, 123);
//
//		for (int i = 0; i < 123; i++)
//			EXPECT_EQ(loaded[i], i);
//	}
//	TEST(TestSerialization, add)
//	{
//		SerializedObject so1;
//		SerializedObject so2;
//
//		so1.save<int>(1);
//		so2.save<double>(2.0);
//		EXPECT_EQ(so1.size(), static_cast<size_t>(sizeof(int)));
//		EXPECT_EQ(so2.size(), static_cast<size_t>(sizeof(double)));
//
//		SerializedObject so3 = so1 + so2;
//		so3.setPosition(0);
//		EXPECT_EQ(so3.size(), static_cast<size_t>(sizeof(int) + sizeof(double)));
//		EXPECT_EQ(so3.load<int>(), 1);
//		EXPECT_EQ(so3.load<double>(), 2.0);
//	}
//	TEST(TestSerialization, save_to_file)
//	{
//		SerializedObject so;
//
//		so.save<char>(110);
//		so.save<short>(13210);
//		so.save<int>(2341230);
//		so.save<int64_t>(11234123412310ll);
//		so.save<float>(1.34123f);
//		so.save<double>(1.1234123412);
//
//		so.saveToFile("testowy.bin");
//
//		SerializedObject loaded("testowy.bin");
//
//		EXPECT_EQ(loaded.load<char>(), 110);
//		EXPECT_EQ(loaded.load<short>(), 13210);
//		EXPECT_EQ(loaded.load<int>(), 2341230);
//		EXPECT_EQ(loaded.load<int64_t>(), 11234123412310ll);
//		EXPECT_EQ(loaded.load<float>(), 1.34123f);
//		EXPECT_EQ(loaded.load<double>(), 1.1234123412);
//	}
//	TEST(TestSerialization, string)
//	{
//		std::string str = "asdf";
//		EXPECT_EQ(serializedSize(str), 8ull);
//
//		SerializedObject so = serialize(str);
//		so.setPosition(0);
//
//		std::string loaded = unserialize<std::string>(so);
//		EXPECT_EQ(loaded, str);
//	}

} /* namespace ml */
