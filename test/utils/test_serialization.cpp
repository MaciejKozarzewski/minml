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
	TEST(TestSerialization, init)
	{
		SerializedObject so1;
		EXPECT_EQ(so1.size(), 0ull);
		EXPECT_EQ(so1.capacity(), 0ull);

		SerializedObject so2(100);
		EXPECT_EQ(so2.size(), 0ull);
		EXPECT_EQ(so2.capacity(), 100ull);
	}
	TEST(TestSerialization, save_load)
	{
		SerializedObject so;

		so.save<char>(110);
		EXPECT_EQ(so.size(), 1ull);
		so.save<short>(13210);
		EXPECT_EQ(so.size(), 3ull);
		so.save<int>(2341230);
		EXPECT_EQ(so.size(), 7ull);
		so.save<int64_t>(11234123412310ll);
		EXPECT_EQ(so.size(), 15ull);
		so.save<float>(1.34123f);
		EXPECT_EQ(so.size(), 19ull);
		so.save<double>(1.1234123412);
		EXPECT_EQ(so.size(), 27ull);

		EXPECT_EQ(so.load<char>(0), 110);
		EXPECT_EQ(so.load<short>(1), 13210);
		EXPECT_EQ(so.load<int>(3), 2341230);
		EXPECT_EQ(so.load<int64_t>(7), 11234123412310ll);
		EXPECT_EQ(so.load<float>(15), 1.34123f);
		EXPECT_EQ(so.load<double>(19), 1.1234123412);
	}
	TEST(TestSerialization, save_load_array)
	{
		SerializedObject so(1000);

		int tmp[123];
		for (int i = 0; i < 123; i++)
			tmp[i] = i;

		const size_t size_in_bytes = sizeof(int) * 123;
		so.save(tmp, size_in_bytes);
		EXPECT_EQ(so.size(), size_in_bytes);

		int loaded[123];
		so.load(loaded, 0, size_in_bytes);

		for (int i = 0; i < 123; i++)
			EXPECT_EQ(loaded[i], i);
	}
	TEST(TestSerialization, save_to_file)
	{
		SerializedObject so;

		so.save<char>(110);
		so.save<short>(13210);
		so.save<int>(2341230);
		so.save<int64_t>(11234123412310ll);
		so.save<float>(1.34123f);
		so.save<double>(1.1234123412);

		so.saveToFile("testowy.bin");

		SerializedObject loaded("testowy.bin");

		EXPECT_EQ(loaded.load<char>(0), 110);
		EXPECT_EQ(loaded.load<short>(1), 13210);
		EXPECT_EQ(loaded.load<int>(3), 2341230);
		EXPECT_EQ(loaded.load<int64_t>(7), 11234123412310ll);
		EXPECT_EQ(loaded.load<float>(15), 1.34123f);
		EXPECT_EQ(loaded.load<double>(19), 1.1234123412);
	}

} /* namespace ml */
