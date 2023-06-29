/*
 * test_nonlinearity.cpp
 *
 *  Created on: Mar 9, 2021
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/ml_memory.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>

#include <iostream>

namespace ml
{

	TEST(TestSoftmax, ForwardOnCPU_fp32)
	{
		Context context;
		Tensor input = toTensor( { { 0.1f, -0.9f, 2.0f, 0.0f }, { 0.3f, -1.0f, 0.7f, -0.1f } });
		Tensor correct_output = toTensor( { { 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f }, { 0.29114823f, 0.07934714f, 0.43434212f,
				0.19516249f } });

		activationForward(context, input, input, ActivationType::SOFTMAX);
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-4f);
	}
	TEST(TestSoftmax, ForwardOnCPU_fp16)
	{
		Context context;
		Tensor input = toTensor( { { 0.1f, -0.9f, 2.0f, 0.0f }, { 0.3f, -1.0f, 0.7f, -0.1f } });
		Tensor correct_output = toTensor( { { 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f }, { 0.29114823f, 0.07934714f, 0.43434212f,
				0.19516249f } });

		input.convertTo(context, DataType::FLOAT16);
		correct_output.convertTo(context, DataType::FLOAT16);

		activationForward(Context(), input, input, ActivationType::SOFTMAX);
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-3f);
	}

	TEST(TestSoftmax, ForwardOnCUDA_fp32)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP_("No CUDA enabled devices");
		Context context(Device::cuda(0));
		Tensor input = toTensor( { { 0.1f, -0.9f, 2.0f, 0.0f }, { 0.3f, -1.0f, 0.7f, -0.1f } });
		Tensor correct_output = toTensor( { { 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f }, { 0.29114823f, 0.07934714f, 0.43434212f,
				0.19516249f } });

		input.moveTo(context.device());

		activationForward(context, input, input, ActivationType::SOFTMAX);
		context.synchronize();
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-4f);
	}
	TEST(TestSoftmax, ForwardOnCUDA_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP_("No CUDA enabled devices");
		Context context(Device::cuda(0));
		Tensor input = toTensor( { { 0.1f, -0.9f, 2.0f, 0.0f }, { 0.3f, -1.0f, 0.7f, -0.1f } });
		Tensor correct_output = toTensor( { { 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f }, { 0.29114823f, 0.07934714f, 0.43434212f,
				0.19516249f } });

		input.moveTo(context.device());
		input.convertTo(context, DataType::FLOAT16);
		correct_output.convertTo(Context(), DataType::FLOAT16);

		activationForward(context, input, input, ActivationType::SOFTMAX);
		context.synchronize();
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-3f);
	}

} /* namespace ml */

