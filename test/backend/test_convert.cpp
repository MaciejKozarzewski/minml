/*
 * test_convert.cpp
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>

#include <gtest/gtest.h>

namespace
{
	template<typename T>
	void transpose(void *dst, const void *src, int dim0, int dim1, int dim2)
	{
		for (int i = 0; i < dim0; i++)
			for (int j = 0; j < dim1; j++)
				for (int k = 0; k < dim2; k++)
				{
					const int src_idx = (i * dim1 + j) * dim2 + k;
					const int dst_idx = (i * dim2 + k) * dim1 + j;
					reinterpret_cast<T*>(dst)[dst_idx] = reinterpret_cast<const T*>(src)[src_idx];
				}
	}

	void unpack_input_fp32(float *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = (mask & 1u) ? 1.0f : 0.0f;
		}
	}
	void unpack_input_fp16(uint16_t *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = (mask & 1u) ? 0x3c00 : 0x000;
		}
	}

//	ml::Shape transpose(const ml::Shape &shape)
//	{
//		return ml::Shape( { shape[0], shape[2], shape[1] });
//	}

}
namespace ml
{

	TEST(TestUnpackInput, cpu_fp32)
	{
		Context context;
		Tensor input(Shape( { 11, 12, 13, 1 }), DataType::INT32, Device::cpu());
		for (int i = 0; i < input.volume(); i++)
			reinterpret_cast<int32_t*>(input.data())[i] = i;

//		Tensor output(Shape( { 11, 12, 13, 32 }), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
	}

//	TEST(TestTranspose021, cpu_fp32)
//	{
//		Context context;
//		Tensor input(Shape( { 14, 76, 45 }), DataType::FLOAT32, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor output(transpose(input.shape()), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//	TEST(TestTranspose021, cpu_fp16)
//	{
//		Context context;
//		Tensor input(Shape( { 14, 76, 45 }), DataType::FLOAT16, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor output(transpose(input.shape()), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint16_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//
//	TEST(TestTranspose021, cuda_fp32)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Context context(Device::cuda(0));
//		Tensor input(Shape( { 14, 176, 145 }), DataType::FLOAT32, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor correct_output(transpose(input.shape()), input.dtype(), Device::cpu());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		input.moveTo(Device::cuda(0));
//		Tensor output(transpose(input.shape()), input.dtype(), Device::cuda(0));
//		transpose_021(context, input, output);
//		context.synchronize();
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//	TEST(TestTranspose021, cuda_fp16)
//	{
//		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
//			GTEST_SKIP();
//		Context context(Device::cuda(0));
//		Tensor input(Shape( { 14, 176, 145 }), DataType::FLOAT16, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor correct_output(transpose(input.shape()), input.dtype(), Device::cpu());
//
//		transpose<uint16_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		input.moveTo(Device::cuda(0));
//		Tensor output(transpose(input.shape()), input.dtype(), Device::cuda(0));
//		transpose_021(context, input, output);
//		context.synchronize();
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}

}
