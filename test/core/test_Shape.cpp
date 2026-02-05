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
	TEST(TestShape, remove_dim)
	{
		Shape s1( { 4, 5, 6, 7 });
		s1.removeDim(0);
		EXPECT_EQ(s1.rank(), 3);
		EXPECT_EQ(s1[0], 5);
		EXPECT_EQ(s1[1], 6);
		EXPECT_EQ(s1[2], 7);

		Shape s2( { 4, 5, 6, 7 });
		s2.removeDim(1);
		EXPECT_EQ(s2.rank(), 3);
		EXPECT_EQ(s2[0], 4);
		EXPECT_EQ(s2[1], 6);
		EXPECT_EQ(s2[2], 7);

		Shape s3( { 4, 5, 6, 7 });
		s3.removeDim(3);
		EXPECT_EQ(s3.rank(), 3);
		EXPECT_EQ(s3[0], 4);
		EXPECT_EQ(s3[1], 5);
		EXPECT_EQ(s3[2], 6);

		Shape s4( { 4 });
		s4.removeDim(0);
		EXPECT_EQ(s4.rank(), 0);
	}
	TEST(TestShape, insert_dim)
	{
		Shape s1( { 4, 5, 6, 7 });
		s1.insertDim(0, 1);
		EXPECT_EQ(s1.rank(), 5);
		EXPECT_EQ(s1[0], 1);
		EXPECT_EQ(s1[1], 4);
		EXPECT_EQ(s1[2], 5);
		EXPECT_EQ(s1[3], 6);
		EXPECT_EQ(s1[4], 7);

		Shape s2( { 4, 7 });
		s2.insertDim(1, 1);
		EXPECT_EQ(s2.rank(), 3);
		EXPECT_EQ(s2[0], 4);
		EXPECT_EQ(s2[1], 1);
		EXPECT_EQ(s2[2], 7);

		Shape s3( { 4, 5, 6, 7 });
		s3.insertDim(4, 1);
		EXPECT_EQ(s3.rank(), 5);
		EXPECT_EQ(s3[0], 4);
		EXPECT_EQ(s3[1], 5);
		EXPECT_EQ(s3[2], 6);
		EXPECT_EQ(s3[3], 7);
		EXPECT_EQ(s3[4], 1);

		Shape s4;
		s4.insertDim(0, 1);
		EXPECT_EQ(s4.rank(), 1);
		EXPECT_EQ(s4[0], 1);
	}
	TEST(TestShape, squeeze)
	{
		Shape s1( { 4, 5, 6, 7 });
		s1.squeeze();
		EXPECT_EQ(s1.rank(), 4);
		EXPECT_EQ(s1[0], 4);
		EXPECT_EQ(s1[1], 5);
		EXPECT_EQ(s1[2], 6);
		EXPECT_EQ(s1[3], 7);

		Shape s2( { 1, 5, 1, 6, 1 });
		s2.squeeze();
		EXPECT_EQ(s2.rank(), 2);
		EXPECT_EQ(s2[0], 5);
		EXPECT_EQ(s2[1], 6);

		Shape s3( { 1, 1, 1, 1, 1, 1 });
		s3.squeeze();
		EXPECT_EQ(s3.rank(), 0);

		Shape s4;
		s4.squeeze();
		EXPECT_EQ(s4.rank(), 0);
	}
	TEST(TestShape, flatten)
	{
		Shape s1( { 4, 5, 6, 7 });
		s1.flatten();
		EXPECT_EQ(s1.rank(), 1);
		EXPECT_EQ(s1[0], 4 * 5 * 6 * 7);

		Shape s2;
		s2.flatten();
		EXPECT_EQ(s2.rank(), 0);

		Shape s3( { 4, 5, 6, 7 });
		s3.flatten( { 0, 1 });
		EXPECT_EQ(s3.rank(), 3);
		EXPECT_EQ(s3[0], 4 * 5);
		EXPECT_EQ(s3[1], 6);
		EXPECT_EQ(s3[2], 7);

		Shape s4( { 4, 5, 6, 7 });
		s4.flatten( { 0, 1, 2 });
		EXPECT_EQ(s4.rank(), 2);
		EXPECT_EQ(s4[0], 4 * 5 * 6);
		EXPECT_EQ(s4[1], 7);

		Shape s5( { 4, 5, 6, 7 });
		s5.flatten( { 2, 3 });
		EXPECT_EQ(s5.rank(), 3);
		EXPECT_EQ(s5[0], 4);
		EXPECT_EQ(s5[1], 5);
		EXPECT_EQ(s5[2], 6 * 7);

		Shape s6( { 4, 5, 6, 7 });
		s6.flatten( { 0 });
		EXPECT_EQ(s6.rank(), 4);
		EXPECT_EQ(s6[0], 4);
		EXPECT_EQ(s6[1], 5);
		EXPECT_EQ(s6[2], 6);
		EXPECT_EQ(s6[3], 7);
	}

} /* namespace ml */

