/*
 * test_gemm_conv.cpp
 *
 *  Created on: Jan 30, 2021
 *      Author: Maciej Kozarzewski
 */

//#include <libml/math/gemm_conv.hpp>
//#include <libml/hardware/DeviceContext.hpp>
//#include <libml/Tensor.hpp>
//#include <libml/Scalar.hpp>
//#include <libml/utils/testing_util.hpp>
//
//#include <gtest/gtest.h>
//
//namespace
//{
//	void baseline_receptive_fields(const ml::Tensor &input, ml::Tensor &matrix, const int kernel_size, const int stride, bool invert)
//	{
//		assert(ml::same_device(input, matrix));
//		assert(input.device().isCPU());
//		const int batch = input.shape(0);
//		const int height = input.shape(1);
//		const int width = input.shape(2);
//		const int filters = input.shape(3);
//
//		const int pad = -kernel_size / 2;
//		int tile_idx = 0;
//		for (int b = 0; b < batch; b++)
//			for (int h = stride - 1; h < height; h += stride)
//				for (int w = stride - 1; w < width; w += stride, tile_idx++)
//				{
//					int tmp_idx = 0;
//					if (invert)
//					{
//						for (int i = kernel_size - 1; i >= 0; i--)
//							for (int j = kernel_size - 1; j >= 0; j--)
//							{
//								if ((h + i + pad) >= 0 && (h + i + pad) < height && (w + j + pad) >= 0 && (w + j + pad) < width)
//								{
//									for (int f = 0; f < filters; f++, tmp_idx++)
//										matrix.set(input.get<float>( { b, h + i + pad, w + j + pad, f }), { tile_idx, tmp_idx });
//								}
//								else
//									for (int f = 0; f < filters; f++, tmp_idx++)
//										matrix.set(0.0f, { tile_idx, tmp_idx });
//							}
//					}
//					else
//					{
//						for (int i = 0; i < kernel_size; i++)
//							for (int j = 0; j < kernel_size; j++)
//							{
//								if ((h + i + pad) >= 0 && (h + i + pad) < height && (w + j + pad) >= 0 && (w + j + pad) < width)
//								{
//									for (int f = 0; f < filters; f++, tmp_idx++)
//										matrix.set(input.get<float>( { b, h + i + pad, w + j + pad, f }), { tile_idx, tmp_idx });
//								}
//								else
//									for (int f = 0; f < filters; f++, tmp_idx++)
//										matrix.set(0.0f, { tile_idx, tmp_idx });
//							}
//					}
//				}
//	}
//}
//
//namespace ml
//{
//
//	TEST(TestConv2D, explicitGemm3x3Transform)
//	{
//		const int kernel_size = 3;
//		Shape shape( { 1, 4, 4, 1 });
//		Tensor input(shape, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());
//
//		baseline_receptive_fields(input, correct, kernel_size, 1, false);
//
//		Device::cpu().setNumberOfThreads(1);
//		Tensor matrix(correct.shape(), "float32", Device::cpu());
//		testing::initForTest(matrix, 1.0f);
//		math::createReceptiveFields(DeviceContext(), input, matrix, kernel_size, kernel_size, false);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			input.moveTo(Device::cuda(0));
//			matrix.moveTo(Device::cuda(0));
//			testing::initForTest(matrix, 1.0f);
//			math::createReceptiveFields(DeviceContext(Device::cuda(0)), input, matrix, kernel_size, kernel_size, false);
//			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//		}
//	}
//	TEST(TestConv2D, explicitGemm3x3TransformInvert)
//	{
//		const int kernel_size = 3;
//		Shape shape( { 1, 4, 4, 1 });
//		Tensor input(shape, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());
//
//		baseline_receptive_fields(input, correct, kernel_size, 1, true);
//
//		Device::cpu().setNumberOfThreads(1);
//		Tensor matrix(correct.shape(), "float32", Device::cpu());
//		testing::initForTest(matrix, 1.0f);
//		math::createReceptiveFields(DeviceContext(), input, matrix, kernel_size, kernel_size, true);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			input.moveTo(Device::cuda(0));
//			matrix.moveTo(Device::cuda(0));
//			testing::initForTest(matrix, 1.0f);
//			math::createReceptiveFields(DeviceContext(Device::cuda(0)), input, matrix, kernel_size, kernel_size, true);
//			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//		}
//	}
//	TEST(TestConv2D, explicitGemm5x5Transform)
//	{
//		const int kernel_size = 5;
//		Shape shape( { 1, 4, 4, 1 });
//		Tensor input(shape, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());
//
//		baseline_receptive_fields(input, correct, kernel_size, 1, false);
//
//		Device::cpu().setNumberOfThreads(1);
//		Tensor matrix(correct.shape(), "float32", Device::cpu());
//		testing::initForTest(matrix, 1.0f);
//		math::createReceptiveFields(DeviceContext(), input, matrix, kernel_size, kernel_size, false);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			input.moveTo(Device::cuda(0));
//			matrix.moveTo(Device::cuda(0));
//			testing::initForTest(matrix, 1.0f);
//			math::createReceptiveFields(DeviceContext(Device::cuda(0)), input, matrix, kernel_size, kernel_size, false);
//			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//		}
//	}
//	TEST(TestConv2D, explicitGemm5x5TransformInvert)
//	{
//		const int kernel_size = 5;
//		Shape shape( { 1, 4, 4, 1 });
//		Tensor input(shape, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		Tensor correct( { shape.volumeWithoutLastDim(), kernel_size * kernel_size * shape.lastDim() }, DataType::FLOAT32, Device::cpu());
//
//		baseline_receptive_fields(input, correct, kernel_size, 1, true);
//
//		Device::cpu().setNumberOfThreads(1);
//		Tensor matrix(correct.shape(), "float32", Device::cpu());
//		testing::initForTest(matrix, 1.0f);
//		math::createReceptiveFields(DeviceContext(), input, matrix, kernel_size, kernel_size, true);
//		EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			input.moveTo(Device::cuda(0));
//			matrix.moveTo(Device::cuda(0));
//			testing::initForTest(matrix, 1.0f);
//			math::createReceptiveFields(DeviceContext(Device::cuda(0)), input, matrix, kernel_size, kernel_size, true);
//			EXPECT_EQ(testing::diffForTest(correct, matrix), 0.0f);
//		}
//	}
//
//} /* namespace ml */
