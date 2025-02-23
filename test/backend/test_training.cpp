/*
 * test_training.cpp
 *
 *  Created on: Mar 24, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/random.hpp>

#include <gtest/gtest.h>

namespace ml
{
	TEST(TestTraining, cross_entropy_loss)
	{
		Tensor output( { 12, 34, 56 }, "float32", Device::cpu());
		Tensor target(output.shape(), "float32", Device::cpu());

		for (int i = 0; i < output.volume(); i++)
		{
			reinterpret_cast<float*>(output.data())[i] = randFloat();
			reinterpret_cast<float*>(target.data())[i] = randFloat();
		}

		const float cpu_loss = crossEntropyLoss(Context(), output, target);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			output.moveTo(device);
			target.moveTo(device);
			const float device_loss = crossEntropyLoss(context, output, target);
			context.synchronize();

			const float diff = std::abs(cpu_loss - device_loss) / output.volume();
			EXPECT_LE(diff, 1.0e-4f);
		}
	}
	TEST(TestTraining, cross_entropy_gradient)
	{
		Tensor output( { 12, 34, 56 }, "float32", Device::cpu());
		Tensor target(output.shape(), "float32", Device::cpu());

		for (int i = 0; i < output.volume(); i++)
		{
			reinterpret_cast<float*>(output.data())[i] = randFloat();
			reinterpret_cast<float*>(target.data())[i] = randFloat();
		}

		Tensor cpu_gradient(output.shape(), "float32", Device::cpu());
		crossEntropyGradient(Context(), cpu_gradient, output, target, 1.23f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			output.moveTo(device);
			target.moveTo(device);
			Tensor device_gradient(output.shape(), "float32", device);
			crossEntropyGradient(context, device_gradient, output, target, 1.23f);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(cpu_gradient, device_gradient), 1.0e-4f);
		}
	}

	TEST(TestTraining, adam_optimize)
	{
		const Shape shape( { 12, 34, 56 });
		Tensor gradient(shape, "float32", Device::cpu());

		Tensor cpu_weights(shape, "float32", Device::cpu());
		testing::initForTest(cpu_weights, 0.0f);
		Tensor cpu_momentum(shape, "float32", Device::cpu());
		Tensor cpu_variance(shape, "float32", Device::cpu());
		for (int i = 1; i < 1000; i++)
		{
			testing::initForTest(gradient, 1.0f + 0.01f * i);
			radamOptimize(Context(), cpu_weights, gradient, cpu_momentum, cpu_variance, 1.0e-3f, 0.9f, 0.999f, i, 0.0f);
		}

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			gradient.moveTo(device);
			Tensor device_weights(shape, "float32", device);
			testing::initForTest(device_weights, 0.0f);
			Tensor device_momentum(shape, "float32", device);
			Tensor device_variance(shape, "float32", device);
			for (int i = 1; i < 1000; i++)
			{
				testing::initForTest(gradient, 1.0f + 0.01f * i);
				radamOptimize(context, device_weights, gradient, device_momentum, device_variance, 1.0e-3f, 0.9f, 0.999f, i, 0.0f);
				context.synchronize();
			}

			EXPECT_LE(testing::diffForTest(cpu_momentum, device_momentum), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(cpu_variance, device_variance), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(cpu_weights, device_weights), 1.0e-4f);
		}
	}

	TEST(TestTraining, l2_regularization)
	{
		const Shape shape( { 12, 34, 56 });
		Tensor gradient(shape, "float32", Device::cpu());
		testing::initForTest(gradient, 1.0f);

		Tensor cpu_weights(shape, "float32", Device::cpu());
		testing::initForTest(cpu_weights, 0.0f);

		l2Regularization(Context(), gradient, cpu_weights, 0.123f, 0.456f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			gradient.moveTo(device);
			Tensor device_weights(shape, "float32", device);
			testing::initForTest(device_weights, 0.0f);
			l2Regularization(context, gradient, device_weights, 0.123f, 0.456f);

			EXPECT_LE(testing::diffForTest(cpu_weights, device_weights), 1.0e-4f);
		}
	}

} /* namespace ml */
