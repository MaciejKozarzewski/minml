/*
 * test_global_pooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/utils/testing_util.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;
	void baseline_pooling_forward(const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 4);
		assert(output.rank() == 3);
		assert(input.firstDim() == output.firstDim());
		assert(output.dim(1) == 2);
		assert(input.lastDim() == output.lastDim());

		const float inv = 1.0f / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
			{
				float avg_value = 0.0f;
				float max_value = input.get( { i, 0, 0, l });
				for (int j = 0; j < input.dim(1); j++)
					for (int k = 0; k < input.dim(2); k++)
					{
						const float x = input.get( { i, j, k, l });
						avg_value += x;
						max_value = std::max(max_value, x);
					}
				output.set(avg_value * inv, { i, 0, l });
				output.set(max_value, { i, 1, l });
			}
	}
	void baseline_pooling_backward(Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input, const Tensor &output)
	{
		assert(input.shape() == gradient_prev.shape());
		assert(output.shape() == gradient_next.shape());
		assert(input.rank() == 4);
		assert(output.rank() == 3);
		assert(input.firstDim() == output.firstDim());
		assert(output.dim(1) == 2);
		assert(input.lastDim() == output.lastDim());

		const float inv = 1.0f / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
					{
						const float max_value = output.get( { i, 1, l });

						const float grad_avg = gradient_next.get( { i, 0, l });
						const float grad_max = (input.get( { i, j, k, l }) == max_value) ? gradient_next.get( { i, 1, l }) : 0.0f;

						const float tmp = gradient_prev.get( { i, j, k, l }) + inv * grad_avg + grad_max;
						gradient_prev.set(tmp, { i, j, k, l });
					}
	}

	void baseline_broadcasting_forward(const Tensor &input, Tensor &output, const Tensor &bias, ActivationType act)
	{
		assert(input.rank() == 4);
		assert(input.shape() == output.shape());
		assert(bias.rank() == 2);
		assert(bias.firstDim() == output.firstDim());
		assert(bias.lastDim() == output.lastDim());

		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
					{
						const float tmp = input.get( { i, j, k, l }) + bias.get( { i, l });
						output.set(tmp, { i, j, k, l });
					}
		activationForward(Context(), output, output, act);
	}
	void baseline_broadcasting_backward(Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, ActivationType act)
	{
		assert(output.rank() == 4);
		assert(gradient_next.shape() == output.shape());
		assert(gradient_prev.rank() == 2);
		assert(gradient_prev.firstDim() == output.firstDim());
		assert(gradient_prev.lastDim() == output.lastDim());

		activationBackward(Context(), gradient_next, gradient_next, output, act);
		for (int i = 0; i < gradient_next.dim(0); i++)
			for (int l = 0; l < gradient_next.dim(3); l++)
			{
				float grad = 0.0f;
				for (int j = 0; j < gradient_next.dim(1); j++)
					for (int k = 0; k < gradient_next.dim(2); k++)
						grad += gradient_next.get( { i, j, k, l });
				gradient_prev.set(grad, { i, l });
			}
	}
}

namespace ml
{
	TEST(TestGlobalPooling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 134 }, "float32", Device::cpu());
		Tensor output( { input.firstDim(), 2, input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output(output.shape(), input.dtype(), Device::cpu());

		testing::initForTest(input, 0.0);

		baseline_pooling_forward(input, correct_output);
		globalAvgAndMaxPoolingForward(context, input, output);

		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.57f);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAvgAndMaxPoolingForward(context, input, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestGlobalPooling, backward)
	{
		Context context;
		Tensor input( { 12, 13, 14, 34 }, "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor output( { input.firstDim(), 2, input.lastDim() }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());

		Tensor correct_gradient_prev(gradient_prev.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0);
		testing::initForTest(gradient_next, 1.57);

		baseline_pooling_forward(input, output);
		baseline_pooling_backward(correct_gradient_prev, gradient_next, input, output);

		output.zeroall();
		globalAvgAndMaxPoolingForward(context, input, output);
		globalAvgAndMaxPoolingBackward(context, gradient_prev, gradient_next, input, output);

		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			output.zeroall();
			globalAvgAndMaxPoolingForward(context, input, output);
			globalAvgAndMaxPoolingBackward(context, gradient_prev, gradient_next, input, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
		}
	}

	TEST(TestGlobalBroadcasting, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 34 }, "float32", Device::cpu());
		Tensor bias( { input.firstDim(), input.lastDim() }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());
		Tensor correct_output(output.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0);
		testing::initForTest(bias, 1.0);

		baseline_broadcasting_forward(input, correct_output, bias, ActivationType::SIGMOID);

		globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);

		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			bias.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
		}
	}
	TEST(TestGlobalBroadcasting, backward)
	{
		Context context;
		Tensor output( { 12, 13, 14, 34 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor gradient_prev( { output.firstDim(), output.lastDim() }, "float32", Device::cpu());
		Tensor correct_gradient_prev(gradient_prev.shape(), "float32", Device::cpu());

		testing::initForTest(gradient_next, 0.0);
		testing::initForTest(output, 0.0);

		baseline_broadcasting_backward(correct_gradient_prev, gradient_next, output, ActivationType::SIGMOID);

		testing::initForTest(gradient_next, 0.0);
		globalBroadcastingBackward(context, gradient_prev, gradient_next, output, ActivationType::SIGMOID);

		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			output.moveTo(device);
			testing::initForTest(gradient_next, 0.0);
			gradient_next.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_prev.zeroall();
			globalBroadcastingBackward(context, gradient_prev, gradient_next, output, ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}

	TEST(TestGlobalPooling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 34 }, "float16", Device::cpu());
		Tensor output( { input.firstDim(), 2, input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output(output.shape(), input.dtype(), Device::cpu());

		testing::initForTest(input, 0.0);

		baseline_pooling_forward(input, correct_output);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			globalAvgAndMaxPoolingForward(context, input, output);
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.57f);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAvgAndMaxPoolingForward(context, input, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}
	}
	TEST(TestGlobalBroadcasting, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 34 }, "float16", Device::cpu());
		Tensor bias( { input.firstDim(), input.lastDim() }, "float16", Device::cpu());
		Tensor output(input.shape(), "float16", Device::cpu());
		Tensor correct_output(output.shape(), "float16", Device::cpu());

		testing::initForTest(input, 0.0);
		testing::initForTest(bias, 1.0);

		baseline_broadcasting_forward(input, correct_output, bias, ActivationType::SIGMOID);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			bias.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}
	}

} /* namespace ml */

