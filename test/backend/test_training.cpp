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

namespace
{
	using namespace ml;
}

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

		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		Context context(Device::cuda(0));
		output.moveTo(context.device());
		target.moveTo(context.device());
		const float cuda_loss = crossEntropyLoss(context, output, target);
		context.synchronize();

		const float diff = std::abs(cpu_loss - cuda_loss) / output.volume();
		EXPECT_LE(diff, 1.0e-4f);
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

		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		Context context(Device::cuda(0));
		output.moveTo(context.device());
		target.moveTo(context.device());
		Tensor cuda_gradient(output.shape(), "float32", context.device());
		crossEntropyGradient(context, cuda_gradient, output, target, 1.23f);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(cpu_gradient, cuda_gradient), 1.0e-4f);
	}

	TEST(TestTraining, adam_optimize)
	{
		const Shape shape( { 12, 34, 56 });
		Tensor gradient(shape, "float32", Device::cpu());

		Tensor cpu_weights(shape, "float32", Device::cpu());
		testing::initForTest(cpu_weights, 0.0f);
		Tensor cpu_momentum(shape, "float32", Device::cpu());
		Tensor cpu_variance(shape, "float32", Device::cpu());
		for (int i = 0; i < 1000; i++)
		{
			testing::initForTest(gradient, 1.0f + 0.01f * i);
			adamOptimize(Context(), cpu_weights, gradient, cpu_momentum, cpu_variance, 1.0e-3f, 0.9f, 0.999f);
		}

		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();

		Context context(Device::cuda(0));
		gradient.moveTo(context.device());
		Tensor cuda_weights(shape, "float32", context.device());
		testing::initForTest(cuda_weights, 0.0f);
		Tensor cuda_momentum(shape, "float32", context.device());
		Tensor cuda_variance(shape, "float32", context.device());
		for (int i = 0; i < 1000; i++)
		{
			testing::initForTest(gradient, 1.0f + 0.01f * i);
			adamOptimize(context, cuda_weights, gradient, cuda_momentum, cuda_variance, 1.0e-3f, 0.9f, 0.999f);
			context.synchronize();
		}

		EXPECT_LE(testing::diffForTest(cpu_momentum, cuda_momentum), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(cpu_variance, cuda_variance), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(cpu_weights, cuda_weights), 1.0e-4f);
	}

	TEST(TestTraining, l2_regularization)
	{
		const Shape shape( { 12, 34, 56 });
		Tensor gradient(shape, "float32", Device::cpu());
		testing::initForTest(gradient, 1.0f);

		Tensor cpu_weights(shape, "float32", Device::cpu());
		testing::initForTest(cpu_weights, 0.0f);

		l2Regularization(Context(), gradient, cpu_weights, 0.123f, 0.456f);

		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		Context context(Device::cuda(0));
		gradient.moveTo(context.device());
		Tensor cuda_weights(shape, "float32", context.device());
		testing::initForTest(cuda_weights, 0.0f);
		l2Regularization(context, gradient, cuda_weights, 0.123f, 0.456f);

		EXPECT_LE(testing::diffForTest(cpu_weights, cuda_weights), 1.0e-4f);
	}

} /* namespace ml */
