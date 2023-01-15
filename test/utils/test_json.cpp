/*
 * test_json.cpp
 *
 *  Created on: Oct 5, 2020
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <minml/utils/json.hpp>

namespace ml
{
	TEST(TestJson, init_null)
	{
		Json j;
		EXPECT_TRUE(j.isNull());
		EXPECT_TRUE(j.isEmpty());
		EXPECT_EQ(j.size(), 0);
	}
	TEST(TestJson, init_bool)
	{
		Json j(true);
		EXPECT_TRUE(j.isBool());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
		EXPECT_EQ(static_cast<bool>(j), true);
	}
	TEST(TestJson, init_int)
	{
		Json j(1234);
		EXPECT_TRUE(j.isNumber());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
		EXPECT_EQ(static_cast<int>(j), 1234);
	}
	TEST(TestJson, init_double)
	{
		Json j(1234.01);
		EXPECT_TRUE(j.isNumber());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
		EXPECT_EQ(static_cast<double>(j), 1234.01);
	}
	TEST(TestJson, init_string)
	{
		Json j("text");
		Json j2(std::string("text"));
		EXPECT_TRUE(j.isString());
		EXPECT_TRUE(j2.isString());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
		EXPECT_EQ(static_cast<std::string>(j), "text");
		EXPECT_EQ(static_cast<std::string>(j2), "text");
	}

	TEST(TestJson, init_array)
	{
		Json j( { 0, 1, 2, 3 });
		EXPECT_TRUE(j.isArray());
		EXPECT_EQ(j.size(), 4);

		j.clear();
		EXPECT_TRUE(j.isArray());
		EXPECT_TRUE(j.isEmpty());
		EXPECT_EQ(j.size(), 0);
	}
	TEST(TestJson, init_array_from_null)
	{
		Json j;
		j[0] = 1;
		EXPECT_TRUE(j.isArray());
		EXPECT_EQ(j.size(), 1);

		j[3] = 3;
		EXPECT_EQ(j.size(), 4);
	}
	TEST(TestJson, array_access)
	{
		Json j( { 0, 1, 2, 3, 4, 5 });
		EXPECT_TRUE(j.isArray());
		EXPECT_EQ(j.size(), 6);

		EXPECT_EQ(static_cast<int>(j[0]), 0);
		EXPECT_TRUE(j[0].isNumber());
		EXPECT_EQ(static_cast<int>(j[5]), 5);

		EXPECT_TRUE(j[6].isNull());
		EXPECT_EQ(j.size(), 7);
	}
	TEST(TestJson, array_modify)
	{
		Json j( { 0, 1, 2, 3, 4, 5 });
		EXPECT_TRUE(j.isArray());
		EXPECT_EQ(j.size(), 6);

		j[0] = "string";
		EXPECT_TRUE(j[0].isString());
	}

	TEST(TestJson, init_object)
	{
		Json j( { { "key", 0.0 } });
		EXPECT_TRUE(j.isObject());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
	}
	TEST(TestJson, init_object_from_null)
	{
		Json j;
		j["key"] = 1;
		EXPECT_TRUE(j.isObject());
		EXPECT_FALSE(j.isEmpty());
		EXPECT_EQ(j.size(), 1);
	}

	TEST(TestJson, dump_null)
	{
		Json j;
		EXPECT_EQ(j.dump(), "null");
	}
	TEST(TestJson, dump_bool)
	{
		Json j(true);
		Json j2(false);
		EXPECT_EQ(j.dump(), "true");
		EXPECT_EQ(j2.dump(), "false");
	}
	TEST(TestJson, dump_int)
	{
		Json j(12345);
		EXPECT_EQ(j.dump(), "12345");
	}
	TEST(TestJson, dump_double)
	{
		Json j(12.340);
		EXPECT_EQ(j.dump(), "12.34");
		j = 12.0;
		EXPECT_EQ(j.dump(), "12");
	}
	TEST(TestJson, dump_string)
	{
		Json j("text");
		EXPECT_EQ(j.dump(), "\"text\"");
	}
	TEST(TestJson, dump_empty_array)
	{
		Json j( { 0 });
		j.clear();
		EXPECT_EQ(j.dump(), "[]");
	}
	TEST(TestJson, dump_int_array)
	{
		Json j( { 0, 1, 2, 3, 4, 5 });
		EXPECT_EQ(j.dump(), "[0,1,2,3,4,5]");
		EXPECT_EQ(j.dump(0), "[0, 1, 2, 3, 4, 5]");
	}
	TEST(TestJson, dump_complex_array)
	{
		Json j( { { "text", 1 }, 2, true, 4 });
		EXPECT_EQ(j.dump(), "[[\"text\",1],2,true,4]");
		EXPECT_EQ(j.dump(0), "[\n[\"text\", 1],\n2,\ntrue,\n4\n]");
		EXPECT_EQ(j.dump(2), "[\n  [\"text\", 1],\n  2,\n  true,\n  4\n]");
	}
	TEST(TestJson, dump_empty_object)
	{
		Json j( { { "key", 0 } });
		j.clear();
		EXPECT_EQ(j.dump(), "{}");
	}
	TEST(TestJson, dump_object)
	{
		Json j( { { "text", 1 }, { "key", true } });
		EXPECT_EQ(j.dump(), "{\"text\":1,\"key\":true}");
		EXPECT_EQ(j.dump(0), "{\n\"text\": 1,\n\"key\": true\n}");
		EXPECT_EQ(j.dump(2), "{\n  \"text\": 1,\n  \"key\": true\n}");
	}

	TEST(TestJson, load_null)
	{
		Json j = Json::load("null");
		EXPECT_TRUE(j.isNull());
	}
	TEST(TestJson, load_bool)
	{
		Json j = Json::load("true");
		EXPECT_TRUE(j.isBool());
		EXPECT_EQ(static_cast<bool>(j), true);

		j = Json::load("false");
		EXPECT_TRUE(j.isBool());
		EXPECT_EQ(static_cast<bool>(j), false);
	}
	TEST(TestJson, load_number)
	{
		Json j = Json::load("123.43");
		EXPECT_TRUE(j.isNumber());
		EXPECT_EQ(static_cast<double>(j), 123.43);
	}
	TEST(TestJson, load_string)
	{
		Json j = Json::load("\"testowy napis\"");
		EXPECT_TRUE(j.isString());
		EXPECT_EQ(static_cast<std::string>(j), "testowy napis");
	}
	TEST(TestJson, load_array)
	{
		Json j = Json::load("[123, true, null, 0.0]");
		EXPECT_TRUE(j.isArray());

		EXPECT_TRUE(j[0].isNumber());
		EXPECT_TRUE(j[1].isBool());
		EXPECT_TRUE(j[2].isNull());
		EXPECT_TRUE(j[3].isNumber());
		EXPECT_EQ(static_cast<double>(j[0]), 123);
		EXPECT_EQ(static_cast<bool>(j[1]), true);
		EXPECT_EQ(static_cast<double>(j[3]), 0.0);
	}
	TEST(TestJson, load_object)
	{
		Json j = Json::load("{\"text\":1,\"key\":true}");
		EXPECT_TRUE(j.isObject());

		EXPECT_TRUE(j["text"].isNumber());
		EXPECT_TRUE(j["key"].isBool());
		EXPECT_EQ(static_cast<double>(j["text"]), 1);
		EXPECT_EQ(static_cast<bool>(j["key"]), true);
	}

} /* namespace ml */

