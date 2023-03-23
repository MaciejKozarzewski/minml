/*
 * test_batchnorm.cpp
 *
 *  Created on: Jan 6, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/utils/testing_util.hpp>

#include <math.h>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	void baseline_forward(const Tensor &input, Tensor &output, const Tensor &weight, Tensor &average, Tensor &stddev, ActivationType act,
			float epsilon = 1.0e-6)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();

		average.zeroall(Context());
		stddev.zeroall(Context());

		for (int b = 0; b < first_dim; b++) // calculate average
			for (int f = 0; f < last_dim; f++)
				average.set(average.get( { f }) + input.get( { b, f }), { f });

		for (int f = 0; f < last_dim; f++) //divide by first dim (in this case batch size)
			average.set(average.get( { f }) / first_dim, { f });

		for (int b = 0; b < first_dim; b++) // subtract average, also calculate variance
			for (int f = 0; f < last_dim; f++)
			{
				float tmp = input.get( { b, f }) - average.get( { f });
				stddev.set(stddev.get( { f }) + tmp * tmp, { f });
			}

		for (int f = 0; f < last_dim; f++) // divide by first dim (in this case batch size)
			stddev.set(stddev.get( { f }) / (first_dim - 1), { f });

		for (int b = 0; b < first_dim; b++) // apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				const float gamma = weight.get( { 2, f });
				const float beta = weight.get( { 3, f });
				const float tmp = (input.get( { b, f }) - average.get( { f })) / std::sqrt(epsilon + stddev.get( { f }));
				output.set(tmp * gamma + beta, { b, f });
			}
		activationForward(Context(), output, output, act);
	}
	void baseline_inference(const Tensor &input, Tensor &output, const Tensor &weight, ActivationType act, float epsilon = 1.0e-6)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const float var = weight.get( { 1, j });
				const float gamma = weight.get( { 2, j });
				const float avg = weight.get( { 0, j });
				const float beta = weight.get( { 3, j });
				const float tmp = gamma * (input.get( { i, j }) - avg) / std::sqrt(epsilon + var) + beta;
				output.set(tmp, { i, j });
			}
		activationForward(Context(), output, output, act);
	}
	void baseline_backward(const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weight,
			Tensor &average, Tensor &variance, ActivationType act, float epsilon = 1.0e-6f)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		Tensor d_sigma( { last_dim }, DataType::FLOAT32, Device::cpu());
		Tensor d_mu( { last_dim }, DataType::FLOAT32, Device::cpu());

		activationBackward(Context(), gradient_next, gradient_next, output, act);
		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				float gamma = 1.0e-16f + weight.get( { 2, f });
				float avg = average.get( { f });
				float var = sqrt(epsilon + variance.get( { f }));
				float in = (input.get( { b, f }) - avg) / var;
				float tmp = -gamma * gradient_next.get( { b, f }) * in / var;
				d_sigma.set(d_sigma.get( { f }) + tmp, { f });

				tmp = -gamma * gradient_next.get( { b, f }) / var;
				d_mu.set(d_mu.get( { f }) + tmp, { f });
			}

		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				float gamma = 1.0e-8f + weight.get( { 2, f });
				float avg = average.get( { f });
				float var = std::sqrt(epsilon + variance.get( { f }));
				float in = (input.get( { b, f }) - avg) / var;
				float m = first_dim;
				float tmp1 = gamma * gradient_next.get( { b, f }) / var;
				float tmp2 = d_sigma.get( { f }) * in / m;
				float tmp3 = d_mu.get( { f }) / m;
				gradient_prev.set(tmp1 + tmp2 + tmp3, { b, f });
			}
	}
	void baseline_update(const Tensor &input, const Tensor &gradient_next, const Tensor &average, const Tensor &variance, Tensor &weight_update,
			float epsilon = 1.0e-6f)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		Tensor d_gamma( { last_dim }, DataType::FLOAT32, Device::cpu());
		Tensor d_beta( { last_dim }, DataType::FLOAT32, Device::cpu());

		for (int b = 0; b < first_dim; b++)
			for (int f = 0; f < last_dim; f++)
			{
				float avg = average.get( { f });
				float var = std::sqrt(epsilon + variance.get( { f }));
				float gamma_update = gradient_next.get( { b, f }) * (input.get( { b, f }) - avg) / var;
				float beta_update = gradient_next.get( { b, f });
				d_gamma.set(d_gamma.get( { f }) + gamma_update, { f });
				d_beta.set(d_beta.get( { f }) + beta_update, { f });
			}
		for (int f = 0; f < last_dim; f++)
		{
			weight_update.set(weight_update.get( { 2, f }) + d_gamma.get( { f }), { 2, f });
			weight_update.set(weight_update.get( { 3, f }) + d_beta.get( { f }), { 3, f });
		}
	}
	void baseline_learn(Tensor &stat, const Tensor &running_stat, int first_dim)
	{
		assert(stat.device().isCPU());
		assert(stat.rank() == 2);
		assert(running_stat.rank() == 2);
		assert(first_dim <= running_stat.dim(0));
		const int last_dim = running_stat.lastDim() / 2;
		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
				stat.set(stat.get( { 0, j }) + running_stat.get( { i, j }), { 0, j });
			for (int j = 0; j < last_dim; j++)
				stat.set(stat.get( { 1, j }) + running_stat.get( { i, 34 + j }), { 1, j });
		}
		for (int j = 0; j < last_dim; j++)
		{
			stat.set(stat.get( { 0, j }) / first_dim, { 0, j });
			stat.set(stat.get( { 1, j }) / first_dim, { 1, j });
		}
	}

	void addScalarToTensor(Tensor &tensor, float scalar)
	{
		for (int i = 0; i < tensor.volume(); i++)
			reinterpret_cast<float*>(tensor.data())[i] += scalar;
	}
}

namespace ml
{
	TEST(TestBatchNorm, forward)
	{
		Context context;

		Tensor input( { 123, 34 }, "float32", Device::cpu());
		Tensor output( { 123, 34 }, "float32", Device::cpu());

		Tensor weight( { 4, 34 }, "float32", Device::cpu());
		Tensor average( { 34 }, "float32", Device::cpu());
		Tensor stddev( { 34 }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(weight, 0.0f);
		addScalarToTensor(weight, 0.001f);

		Tensor correct(output.shape(), "float32", Device::cpu());
		baseline_forward(input, correct, weight, average, stddev, ActivationType::SIGMOID);

		Tensor running_stats( { 64, 2 * 34 }, "float32", Device::cpu());
		batchnormForward(context, input, output, weight, running_stats, 0, ActivationType::SIGMOID);

		{
			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
			Tensor avg = running_stats.view( { 34 }, 0 * 34);
			Tensor dev = running_stats.view( { 34 }, 1 * 34);
			EXPECT_LE(testing::diffForTest(average, avg), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(stddev, dev), 1.0e-4f);
		}

		if (Device::numberOfCudaDevices() > 0)
		{
			Context context(Device::cuda(0));
			input.moveTo(Device::cuda(0));
			output.moveTo(Device::cuda(0));
			weight.moveTo(Device::cuda(0));
			running_stats.moveTo(Device::cuda(0));
			running_stats.zeroall(context);
			output.zeroall(context);

			Tensor avg = running_stats.view( { 34 }, 0 * 34);
			Tensor dev = running_stats.view( { 34 }, 1 * 34);

			batchnormForward(context, input, output, weight, running_stats, 0, ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(average, avg), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(stddev, dev), 1.0e-4f);
		}
	}
	TEST(TestBatchNorm, inference)
	{
		Tensor input( { 123, 34 }, "float32", Device::cpu());
		Tensor output( { 123, 34 }, "float32", Device::cpu());
		Tensor weight( { 4, 34 }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(weight, 0.0f);
		addScalarToTensor(weight, 1.1f);

		Tensor correct(output.shape(), "float32", Device::cpu());
		baseline_inference(input, correct, weight, ActivationType::TANH);

		batchnormInference(Context(), input, output, weight, ActivationType::TANH);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		if (Device::numberOfCudaDevices() > 0)
		{
			Context context(Device::cuda(0));
			input.moveTo(Device::cuda(0));
			output.moveTo(Device::cuda(0));
			weight.moveTo(Device::cuda(0));
			context.synchronize();

			batchnormInference(context, input, output, weight, ActivationType::TANH);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
		}
	}
	TEST(TestBatchNorm, backward)
	{
		Context context;

		Tensor input( { 123, 34 }, "float32", Device::cpu());
		Tensor output( { 123, 34 }, "float32", Device::cpu());
		Tensor gradient_prev( { 123, 34 }, "float32", Device::cpu());
		Tensor gradient_next( { 123, 34 }, "float32", Device::cpu());

		Tensor weight( { 4, 34 }, "float32", Device::cpu());
		Tensor average( { 34 }, "float32", Device::cpu());
		Tensor stddev( { 34 }, "float32", Device::cpu());

		Tensor weight_update( { 4, 34 }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weight, 0.0f);
		addScalarToTensor(weight, 1.1f);

		testing::initForTest(weight_update, 0.1f);

		Tensor correct_prev(output.shape(), "float32", Device::cpu());
		Tensor correct_weight_update( { 4, 34 }, "float32", Device::cpu());
		correct_weight_update.copyFrom(Context(), weight_update);

		baseline_forward(input, output, weight, average, stddev, ActivationType::SIGMOID);
		baseline_backward(input, output, correct_prev, gradient_next, weight, average, stddev, ActivationType::SIGMOID);
		baseline_update(input, gradient_next, average, stddev, correct_weight_update);

		Tensor running_stats( { 64, 2 * 34 }, "float32", Device::cpu());
		output.zeroall(context);
		testing::initForTest(gradient_next, 1.57f);
		batchnormForward(context, input, output, weight, running_stats, 0, ActivationType::SIGMOID);
		batchnormBackward(context, input, output, gradient_prev, gradient_next, weight, weight_update, running_stats, 0, ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (Device::numberOfCudaDevices() > 0)
		{
			Context context(Device::cuda(0));
			testing::initForTest(weight_update, 0.1f);
			testing::initForTest(gradient_next, 1.57f);

			weight_update.moveTo(Device::cuda(0));
			input.moveTo(Device::cuda(0));
			output.moveTo(Device::cuda(0));
			gradient_prev.moveTo(Device::cuda(0));
			gradient_next.moveTo(Device::cuda(0));
			weight.moveTo(Device::cuda(0));
			running_stats.moveTo(Device::cuda(0));

			output.zeroall(context);
			running_stats.zeroall(context);
			gradient_prev.zeroall(context);

			batchnormForward(context, input, output, weight, running_stats, 0, ActivationType::SIGMOID);
			batchnormBackward(context, input, output, gradient_prev, gradient_next, weight, weight_update, running_stats, 0, ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}
	TEST(TestBatchNorm, learn)
	{
		Context context;
		Tensor correct_weights( { 4, 34 }, "float32", Device::cpu());
		Tensor weights( { 4, 34 }, "float32", Device::cpu());
		Tensor running_stat( { 64, 2 * 34 }, "float32", Device::cpu());
		testing::initForTest(running_stat, 0.0);

		baseline_learn(correct_weights, running_stat, 50);

		batchnormUpdate(context, running_stat, 50, weights, true, true);
		EXPECT_LE(testing::diffForTest(correct_weights, weights), 1.0e-4f);

		if (Device::numberOfCudaDevices())
		{
			Context context(Device::cuda(0));
			weights.moveTo(Device::cuda(0));
			running_stat.moveTo(Device::cuda(0));
			weights.zeroall(context);

			batchnormUpdate(context, running_stat, 50, weights, true, true);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_weights, weights), 1.0e-4f);
		}
	}

} /* namespace ml */

