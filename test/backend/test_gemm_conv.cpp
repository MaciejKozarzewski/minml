/*
 * test_gemm_conv.cpp
 *
 *  Created on: Jan 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/testing_util.hpp>

#include <gtest/gtest.h>

namespace
{
	void baseline_receptive_fields(const ml::Tensor &input, ml::Tensor &matrix, const int kernel_size, const int stride, bool invert)
	{
		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters = input.dim(3);

		const int pad = -kernel_size / 2;
		int tile_idx = 0;
		for (int b = 0; b < batch; b++)
			for (int h = stride - 1; h < height; h += stride)
				for (int w = stride - 1; w < width; w += stride, tile_idx++)
				{
					int tmp_idx = 0;
					if (invert)
					{
						for (int i = kernel_size - 1; i >= 0; i--)
							for (int j = kernel_size - 1; j >= 0; j--)
							{
								if ((h + i + pad) >= 0 && (h + i + pad) < height && (w + j + pad) >= 0 && (w + j + pad) < width)
								{
									for (int f = 0; f < filters; f++, tmp_idx++)
										matrix.at( { tile_idx, tmp_idx }) = input.get( { b, h + i + pad, w + j + pad, f });
								}
								else
									for (int f = 0; f < filters; f++, tmp_idx++)
										matrix.at( { tile_idx, tmp_idx }) = 0.0f;
							}
					}
					else
					{
						for (int i = 0; i < kernel_size; i++)
							for (int j = 0; j < kernel_size; j++)
							{
								if ((h + i + pad) >= 0 && (h + i + pad) < height && (w + j + pad) >= 0 && (w + j + pad) < width)
								{
									for (int f = 0; f < filters; f++, tmp_idx++)
										matrix.at( { tile_idx, tmp_idx }) = input.get( { b, h + i + pad, w + j + pad, f });
								}
								else
									for (int f = 0; f < filters; f++, tmp_idx++)
										matrix.at( { tile_idx, tmp_idx }) = 0.0f;
							}
					}
				}
	}
}

namespace ml
{

	TEST(TestConv2D, im2row3x3)
	{
		const int kernel_size = 3;
		Tensor input( { 11, 12, 13, 35 }, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		Tensor correct( { input.shape().volumeWithoutLastDim(), kernel_size * kernel_size * input.lastDim() }, DataType::FLOAT32, Device::cpu());

		baseline_receptive_fields(input, correct, kernel_size, 1, false);

		Tensor matrix = zeros_like(correct);
		testing::initForTest(matrix, 1.0f);
		im2row(Context(), matrix, input, kernel_size, false, nullptr);
		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			matrix.moveTo(device);
			testing::initForTest(matrix, 1.0f);

			im2row(context, matrix, input, kernel_size, false, nullptr);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
		}
	}
	TEST(TestConv2D, im2row3x3_invert)
	{
		const int kernel_size = 3;
		Shape shape( { 11, 12, 13, 35 });
		Tensor input(shape, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());

		baseline_receptive_fields(input, correct, kernel_size, 1, true);

		Tensor matrix(correct.shape(), "float32", Device::cpu());
		testing::initForTest(matrix, 1.0f);
//		im2row(Context(), matrix, input, kernel_size, true, nullptr);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			matrix.moveTo(device);
			testing::initForTest(matrix, 1.0f);

			im2row(context, matrix, input, kernel_size, true, nullptr);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
		}
	}
	TEST(TestConv2D, im2row5x5)
	{
		const int kernel_size = 5;
		Shape shape( { 11, 12, 13, 35 });
		Tensor input(shape, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());

		baseline_receptive_fields(input, correct, kernel_size, 1, false);

		Tensor matrix(correct.shape(), "float32", Device::cpu());
		testing::initForTest(matrix, 1.0f);
		im2row(Context(), matrix, input, kernel_size, false, nullptr);
		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			matrix.moveTo(device);
			testing::initForTest(matrix, 1.0f);

			im2row(context, matrix, input, kernel_size, false, nullptr);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
		}
	}
	TEST(TestConv2D, im2row5x5_invert)
	{
		const int kernel_size = 5;
		Shape shape( { 11, 12, 13, 35 });
		Tensor input(shape, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());

		baseline_receptive_fields(input, correct, kernel_size, 1, true);

		Tensor matrix(correct.shape(), "float32", Device::cpu());
		testing::initForTest(matrix, 1.0f);
//		im2row(Context(), matrix, input, kernel_size, true, nullptr);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			matrix.moveTo(device);
			testing::initForTest(matrix, 1.0f);

			im2row(context, matrix, input, kernel_size, true, nullptr);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
		}
	}

} /* namespace ml */
