/*
 * test_batchnorm.cpp
 *
 *  Created on: Jan 6, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/utils/testing_util.hpp>

#include <math.h>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	void baseline_forward(const Tensor &input, Tensor &output, const Tensor &weight, Tensor &stats, int stat_id, ActivationType act, float epsilon =
			1.0e-6)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();

		for (int f = 0; f < last_dim; f++)
			stats.set(first_dim, { stat_id, 3 * f + 0 });

		for (int b = 0; b < first_dim; b++) // calculate average
			for (int f = 0; f < last_dim; f++)
				stats.set(stats.get( { stat_id, 3 * f + 1 }) + input.get( { b, f }), { stat_id, 3 * f + 1 });

		for (int f = 0; f < last_dim; f++) //divide by first dim (in this case batch size)
			stats.set(stats.get( { stat_id, 3 * f + 1 }) / first_dim, { stat_id, 3 * f + 1 });

		for (int b = 0; b < first_dim; b++) // subtract average, also calculate variance
			for (int f = 0; f < last_dim; f++)
			{
				const float tmp = input.get( { b, f }) - stats.get( { stat_id, 3 * f + 1 });
				stats.set(stats.get( { stat_id, 3 * f + 2 }) + tmp * tmp, { stat_id, 3 * f + 2 });
			}

		for (int b = 0; b < first_dim; b++) // apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				const float avg = stats.get( { stat_id, 3 * f + 1 });
				const float stddev = std::sqrt(epsilon + stats.get( { stat_id, 3 * f + 2 }) / (first_dim - 1));
				const float gamma = weight.get( { 2, f });
				const float beta = weight.get( { 3, f });
				const float tmp = (input.get( { b, f }) - avg) / stddev;
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
			const Tensor &stats, int stat_id, ActivationType act, float epsilon = 1.0e-6f)
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
				const float gamma = 1.0e-16f + weight.get( { 2, f });
				const float avg = stats.get( { stat_id, 3 * f + 1 });
				const float var = sqrt(epsilon + stats.get( { stat_id, 3 * f + 2 }) / (first_dim - 1));
				const float in = (input.get( { b, f }) - avg) / var;
				float tmp = -gamma * gradient_next.get( { b, f }) * in / var;
				d_sigma.set(d_sigma.get( { f }) + tmp, { f });

				tmp = -gamma * gradient_next.get( { b, f }) / var;
				d_mu.set(d_mu.get( { f }) + tmp, { f });
			}

		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				const float gamma = 1.0e-8f + weight.get( { 2, f });
				const float avg = stats.get( { stat_id, 3 * f + 1 });
				const float var = sqrt(epsilon + stats.get( { stat_id, 3 * f + 2 }) / (first_dim - 1));
				const float in = (input.get( { b, f }) - avg) / var;
				const float m = first_dim;
				const float tmp1 = gamma * gradient_next.get( { b, f }) / var;
				const float tmp2 = d_sigma.get( { f }) * in / m;
				const float tmp3 = d_mu.get( { f }) / m;
				gradient_prev.set(tmp1 + tmp2 + tmp3, { b, f });
			}
	}
	void baseline_update(const Tensor &input, const Tensor &gradient_next, const Tensor &stats, int stat_id, Tensor &weight_update, float epsilon =
			1.0e-6f)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		Tensor d_gamma( { last_dim }, DataType::FLOAT32, Device::cpu());
		Tensor d_beta( { last_dim }, DataType::FLOAT32, Device::cpu());

		for (int b = 0; b < first_dim; b++)
			for (int f = 0; f < last_dim; f++)
			{
				const float avg = stats.get( { stat_id, 3 * f + 1 });
				const float var = sqrt(epsilon + stats.get( { stat_id, 3 * f + 2 }) / (first_dim - 1));
				const float gamma_update = gradient_next.get( { b, f }) * (input.get( { b, f }) - avg) / var;
				const float beta_update = gradient_next.get( { b, f });
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
				stat.set(stat.get( { 1, j }) + running_stat.get( { i, last_dim + j }), { 1, j });
		}
		for (int j = 0; j < last_dim; j++)
		{
			stat.set(stat.get( { 0, j }) / first_dim, { 0, j });
			stat.set(stat.get( { 1, j }) / first_dim, { 1, j });
		}
	}

	void add_scalar_to_tensor(Tensor &tensor, float scalar)
	{
		assert(tensor.device().isCPU());
		for (int i = 0; i < tensor.volume(); i++)
			reinterpret_cast<float*>(tensor.data())[i] += scalar;
	}
}

namespace ml
{
	TEST(TestBatchNorm, forward)
	{
		const int batch_size = 2;
		const int height = 3;
		const int width = 4;
		const int filters = 5;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float32", Device::cpu());
		std::vector<Tensor> output;
		for (int i = 0; i < 10; i++)
			output.push_back(Tensor(input.shape(), "float32", Device::cpu()));

		Tensor weight( { 4, filters }, "float32", Device::cpu());
		Tensor correct_stats( { (int) output.size(), 3 * filters }, "float32", Device::cpu());

		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.001f);

		std::vector<Tensor> correct;
		for (size_t i = 0; i < output.size(); i++)
			correct.push_back(Tensor(input.shape(), "float32", Device::cpu()));
		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		for (size_t i = 0; i < output.size(); i++)
		{
			testing::initForTest(input, 0.1f * i);
			baseline_forward(input, correct[i], weight, correct_stats, i, ActivationType::SIGMOID);
			batchnormForward(context, input, output[i], weight, running_stats, i, ActivationType::SIGMOID);

			EXPECT_LE(testing::diffForTest(correct[i], output[i]), 1.0e-4f);
			Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
			Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);
			EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-4f);
		}

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			for (size_t i = 0; i < output.size(); i++)
			{
				output[i].moveTo(device);
				output[i].zeroall();
			}
			weight.moveTo(device);
			running_stats.moveTo(device);
			running_stats.zeroall();

			for (size_t i = 0; i < output.size(); i++)
			{
				testing::initForTest(input, 0.1f * i);
				batchnormForward(context, input, output[i], weight, running_stats, i, ActivationType::SIGMOID);
				context.synchronize();

				EXPECT_LE(testing::diffForTest(correct[i], output[i]), 1.0e-4f);
				Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
				Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);
				EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-4f);
			}
		}
	}
	TEST(TestBatchNorm, inference)
	{
		Tensor input( { 123, 34 }, "float32", Device::cpu());
		Tensor output( { 123, 34 }, "float32", Device::cpu());
		Tensor weight( { 4, 34 }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.1f);

		Tensor correct(output.shape(), "float32", Device::cpu());
		baseline_inference(input, correct, weight, ActivationType::TANH);

		batchnormInference(Context(), input, output, weight, ActivationType::TANH);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weight.moveTo(device);
			output.zeroall();
			context.synchronize();

			batchnormInference(context, input, output, weight, ActivationType::TANH);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
		}
	}
	TEST(TestBatchNorm, backward)
	{
		const int batch_size = 256;
		const int height = 15;
		const int width = 15;
		const int filters = 64;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor gradient_next(input.shape(), "float32", Device::cpu());

		Tensor weight( { 4, filters }, "float32", Device::cpu());
		Tensor stats( { 10, 3 * filters }, "float32", Device::cpu());

		Tensor weight_update( { 4, filters }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.1f);

		testing::initForTest(weight_update, 0.1f);

		Tensor correct_prev(output.shape(), "float32", Device::cpu());
		Tensor correct_weight_update( { 4, filters }, "float32", Device::cpu());
		correct_weight_update.copyFrom(Context(), weight_update);

		const int stat_id = 2;

		baseline_forward(input, output, weight, stats, stat_id, ActivationType::SIGMOID);
		baseline_backward(input, output, correct_prev, gradient_next, weight, stats, stat_id, ActivationType::SIGMOID);
		baseline_update(input, gradient_next, stats, stat_id, correct_weight_update);

		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		output.zeroall();
		testing::initForTest(gradient_next, 1.57f);
		batchnormForward(context, input, output, weight, running_stats, stat_id, ActivationType::SIGMOID);
		batchnormBackward(context, input, output, gradient_prev, gradient_next, weight, weight_update, running_stats, stat_id,
				ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 0.1f);
			testing::initForTest(gradient_next, 1.57f);

			weight_update.moveTo(device);
			input.moveTo(device);
			output.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weight.moveTo(device);
			running_stats.moveTo(device);

			output.zeroall();
			running_stats.zeroall();
			gradient_prev.zeroall();

			batchnormForward(context, input, output, weight, running_stats, stat_id, ActivationType::SIGMOID);
			batchnormBackward(context, input, output, gradient_prev, gradient_next, weight, weight_update, running_stats, stat_id,
					ActivationType::SIGMOID);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}
	TEST(TestBatchNorm, learn)
	{
//		const int filters = 64;
//		Context context;
//
//		Tensor correct_weights( { 4, filters }, "float32", Device::cpu());
//		Tensor weights( { 4, filters }, "float32", Device::cpu());
//		Tensor running_stat( { 64, 3 * filters }, "float32", Device::cpu());
//		testing::initForTest(running_stat, 0.0, 1.1);
//
//		baseline_learn(correct_weights, running_stat, 50);
//
//		batchnormUpdate(context, running_stat, 50, weights, true, true);
//		EXPECT_LE(testing::diffForTest(correct_weights, weights), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices())
//		{
//			Context context(Device::cuda(0));
//			weights.moveTo(Device::cuda(0));
//			running_stat.moveTo(Device::cuda(0));
//			weights.zeroall(context);
//
//			batchnormUpdate(context, running_stat, 50, weights, true, true);
//			context.synchronize();
//			EXPECT_LE(testing::diffForTest(correct_weights, weights), 1.0e-4f);
//		}
	}

} /* namespace ml */

