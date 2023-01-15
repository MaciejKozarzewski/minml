/*
 * test_shape.cpp
 *
 *  Created on: May 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Shape.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/json.hpp>

#include <gtest/gtest.h>

namespace ml
{
	TEST(TestShape, init)
	{
		Shape s1( { });
		EXPECT_EQ(s1.rank(), 0);
		EXPECT_EQ(s1.volume(), 0);

		Shape s2( { 1, 2, 3, 4 });
		EXPECT_EQ(s2.rank(), 4);
		EXPECT_EQ(s2.volume(), 1 * 2 * 3 * 4);
	}
	TEST(TestShape, get)
	{
		Shape s1( { 1, 2, 3, 4 });

		EXPECT_EQ(s1[0], 1);
		EXPECT_EQ(s1[1], 2);
		EXPECT_EQ(s1[2], 3);
		EXPECT_EQ(s1[3], 4);
		EXPECT_THROW(s1[4], IndexOutOfBounds);
	}
	TEST(TestShape, set)
	{
		Shape s1( { 1, 2, 3, 4 });
		Shape s2 = s1;
		EXPECT_TRUE(s1 == s2);
		s2[0] = 0;
		s2 = s1;
		EXPECT_TRUE(s1 == s2);

		Shape s3 = Shape( { 1, 2, 3, 4 });
		EXPECT_TRUE(s1 == s3);
		s3[0] = 0;
		s3 = Shape( { 1, 2, 3 });
	}
	TEST(TestShape, get_dim)
	{
		Shape s1( { });
		Shape s2( { 1 });
		Shape s3( { 1, 2, 3, 4 });

		EXPECT_EQ(s1.firstDim(), 0);
		EXPECT_EQ(s2.firstDim(), 1);
		EXPECT_EQ(s3.firstDim(), 1);

		EXPECT_EQ(s1.lastDim(), 0);
		EXPECT_EQ(s2.lastDim(), 1);
		EXPECT_EQ(s3.lastDim(), 4);
	}
	TEST(TestShape, volume)
	{
		Shape s2( { 4 });
		Shape s3( { 4, 5, 6, 7 });

		EXPECT_EQ(s2.volume(), 4);
		EXPECT_EQ(s2.volumeWithoutFirstDim(), 0);
		EXPECT_EQ(s2.volumeWithoutLastDim(), 0);
		EXPECT_EQ(s2.volume( { 0 }), 4);
		EXPECT_THROW(s2.volume( { 1 }), IndexOutOfBounds);

		EXPECT_EQ(s3.volume(), 4 * 5 * 6 * 7);
		EXPECT_EQ(s3.volumeWithoutFirstDim(), 5 * 6 * 7);
		EXPECT_EQ(s3.volumeWithoutLastDim(), 4 * 5 * 6);
		EXPECT_EQ(s3.volume( { 0 }), 4);
		EXPECT_EQ(s3.volume( { 1, 2 }), 5 * 6);
	}
	TEST(TestShape, serialization)
	{
		Shape shape( { 1, 2, 3, 4 });

		Json j = shape.serialize();
		EXPECT_TRUE(j.isArray());

		Shape loaded(j);
		EXPECT_EQ(shape, loaded);
	}

} /* namespace ml */

