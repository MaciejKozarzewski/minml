/*
 * test_batchnorm.cpp
 *
 *  Created on: Jan 6, 2021
 *      Author: Maciej Kozarzewski
 */

//#include <minml/core/math.hpp>
//#include <minml/core/Context.hpp>
//#include <minml/core/Tensor.hpp>
//#include <minml/utils/testing_util.hpp>
//
//#include <math.h>
//#include <gtest/gtest.h>
//
//namespace
//{
//	using namespace ml;
//
//	void baseline_forward(const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias, Tensor &average, Tensor &stddev,
//			ActivationType act, float epsilon = 1.0e-6)
//	{
//		assert(input.device().isCPU());
//		const int first_dim = input.shape().volumeWithoutLastDim();
//		const int last_dim = input.shape().lastDim();
//
//		average.zeroall(Context());
//		stddev.zeroall(Context());
//
//		for (int b = 0; b < first_dim; b++) //calculate average
//			for (int f = 0; f < last_dim; f++)
//				average.set(average.get( { f }) + input.get( { b, f }), { f });
//
//		for (int f = 0; f < last_dim; f++) //divide by first dim (in this case batch size)
//			average.set(average.get( { f }) / first_dim, { f });
//
//		for (int b = 0; b < first_dim; b++) //subtract average, also calculate variance
//			for (int f = 0; f < last_dim; f++)
//			{
//				float tmp = input.get( { b, f }) - average.get( { f });
//				stddev.set(stddev.get( { f }) + tmp * tmp, { f });
//			}
//
//		for (int f = 0; f < last_dim; f++) //divide by first dim (in this case batch size)
//		{
//			float var = sqrt(1.0e-16 + epsilon + stddev.get( { f }) / first_dim);
//			stddev.set(var, { f });
//		}
//
//		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
//			for (int f = 0; f < last_dim; f++)
//			{
//				float gamma = weight.get( { 1, f });
//				float beta = bias.get( { 1, f });
//				float tmp = (input.get( { b, f }) - average.get( { f })) / stddev.get( { f });
//				output.set(tmp * gamma + beta, { b, f });
//			}
//		activationForwardInPlace(Context(), output, act);
//	}
//	void baseline_inference(const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias, ActivationType act)
//	{
//		assert(input.device().isCPU());
//		const int first_dim = input.shape().volumeWithoutLastDim();
//		const int last_dim = input.shape().lastDim();
//		for (int i = 0; i < first_dim; i++)
//			for (int j = 0; j < last_dim; j++)
//			{
//				float var = weight.get( { 0, j });
//				float gamma = weight.get( { 1, j });
//				float avg = bias.get( { 0, j });
//				float beta = bias.get( { 1, j });
//				float tmp = gamma * (input.get( { i, j }) - avg) / var + beta;
//				output.set(tmp, { i, j });
//			}
//		activationForwardInPlace(Context(), output, act);
//	}
//	void baseline_backward(const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weight,
//			Tensor &average, Tensor &variance, ActivationType act)
//	{
//		assert(input.device().isCPU());
//		const int first_dim = input.shape().volumeWithoutLastDim();
//		const int last_dim = input.shape().lastDim();
//		Tensor d_sigma( { last_dim }, DataType::FLOAT32, Device::cpu());
//		Tensor d_mu( { last_dim }, DataType::FLOAT32, Device::cpu());
//
//		activationBackwardInPlace(Context(), gradient_next, output, act);
//		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
//			for (int f = 0; f < last_dim; f++)
//			{
//				float gamma = 1.0e-16f + weight.get( { 1, f });
//				float avg = average.get( { f });
//				float var = variance.get( { f });
//				float in = (input.get( { b, f }) - avg) / var;
//				float tmp = -gamma * gradient_next.get( { b, f }) * in / var;
//				d_sigma.set(d_sigma.get( { f }) + tmp, { f });
//
//				tmp = -gamma * gradient_next.get( { b, f }) / var;
//				d_mu.set(d_mu.get( { f }) + tmp, { f });
//			}
//
//		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
//			for (int f = 0; f < last_dim; f++)
//			{
//				float gamma = 1.0e-8f + weight.get( { 1, f });
//				float avg = average.get( { f });
//				float var = variance.get( { f });
//				float in = (input.get( { b, f }) - avg) / var;
//				float m = first_dim;
//				float tmp1 = gamma * gradient_next.get( { b, f }) / var;
//				float tmp2 = d_sigma.get( { f }) * in / m;
//				float tmp3 = d_mu.get( { f }) / m;
//				gradient_prev.set(tmp1 + tmp2 + tmp3, { b, f });
//			}
//	}
//	void baseline_update(const Tensor &input, const Tensor &gradient_next, const Tensor &average, const Tensor &variance, Tensor &weight_update,
//			Tensor &bias_update)
//	{
//		assert(input.device().isCPU());
//		const int first_dim = input.shape().volumeWithoutLastDim();
//		const int last_dim = input.shape().lastDim();
//		Tensor d_gamma( { last_dim }, DataType::FLOAT32, Device::cpu());
//		Tensor d_beta( { last_dim }, DataType::FLOAT32, Device::cpu());
//
//		for (int b = 0; b < first_dim; b++)
//			for (int f = 0; f < last_dim; f++)
//			{
//				float avg = average.get( { f });
//				float var = variance.get( { f });
//				float gamma_update = gradient_next.get( { b, f }) * (input.get( { b, f }) - avg) / var;
//				float beta_update = gradient_next.get( { b, f });
//				d_gamma.set(d_gamma.get( { f }) + gamma_update, { f });
//				d_beta.set(d_beta.get( { f }) + beta_update, { f });
//			}
//		for (int f = 0; f < last_dim; f++)
//		{
//			weight_update.set(weight_update.get( { 1, f }) + d_gamma.get( { f }), { 1, f });
//			bias_update.set(bias_update.get( { 1, f }) + d_beta.get( { f }), { 1, f });
//		}
//	}
//	void baseline_learn(Tensor &stat, const Tensor &running_stat, int first_dim)
//	{
//		assert(stat.device().isCPU());
//		assert(stat.rank() == 1);
//		assert(running_stat.rank() == 2);
//		assert(first_dim <= running_stat.dim(0));
//		for (int i = 0; i < stat.dim(0); i++)
//		{
//			float tmp = 0.0f;
//			for (int j = 0; j < first_dim; j++)
//				tmp += running_stat.get( { j, i });
//			stat.set(tmp / first_dim, { i });
//		}
//	}
//}
//
//namespace ml
//{
//	TEST(TestBatchNorm, forward)
//	{
//		// storage + 0 * filters : sqrt(var + epsilon) (store)
//		// storage + 1 * filters : gamma (load)
//		// storage + 2 * filters : avg (store)
//		// storage + 3 * filters : beta (load)
//
//		Tensor input( { 123, 34 }, "float32", Device::cpu());
//		Tensor output( { 123, 34 }, "float32", Device::cpu());
//
//		Tensor weight( { 2, 34 }, "float32", Device::cpu());
//		Tensor bias( { 2, 34 }, "float32", Device::cpu());
//		Tensor average( { 34 }, "float32", Device::cpu());
//		Tensor stddev( { 34 }, "float32", Device::cpu());
//
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(weight, 0.0f);
//		testing::initForTest(bias, 1.57f);
//		math::addScalarToTensor(Context(), weight, 0.001f);
//
//		Tensor correct(output.shape(), "float32", Device::cpu());
//		baseline_forward(input, correct, weight, bias, average, stddev, NonlinearityType::SIGMOID);
//
//		Tensor storage( { 6 + 8, 34 }, "float32", Device::cpu());
//		ml::memcpy(storage.data<float>() + 34, Device::cpu(), weight.data<float>() + 34, Device::cpu(), sizeof(float) * 34); // copy gamma to storage
//		ml::memcpy(storage.data<float>() + 3 * 34, Device::cpu(), bias.data<float>() + 34, Device::cpu(), sizeof(float) * 34); // copy beta to storage
//		math::batchnormForward(Context(), input, output, storage, NonlinearityType::SIGMOID);
//
//		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//		Tensor avg = storage.view( { 34 }, 2 * 34);
//		Tensor dev = storage.view( { 34 }, 0 * 34);
//		EXPECT_LE(testing::diffForTest(average, avg), 1.0e-4f);
//		EXPECT_LE(testing::diffForTest(stddev, dev), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			Context context(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//			weight.moveTo(Device::cuda(0));
//			bias.moveTo(Device::cuda(0));
//			storage.moveTo(Device::cuda(0));
//			avg.zeroall(context);
//			dev.zeroall(context);
//			output.zeroall(context);
//
//			math::batchnormForward(context, input, output, storage, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//			EXPECT_LE(testing::diffForTest(average, avg), 1.0e-4f);
//			EXPECT_LE(testing::diffForTest(stddev, dev), 1.0e-4f);
//		}
//	}
//	TEST(TestBatchNorm, inference)
//	{
//		Tensor input( { 123, 34 }, "float32", Device::cpu());
//		Tensor output( { 123, 34 }, "float32", Device::cpu());
//		Tensor weight( { 2, 34 }, "float32", Device::cpu());
//		Tensor bias( { 2, 34 }, "float32", Device::cpu());
//
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(weight, 0.0f);
//		testing::initForTest(bias, 1.57f);
//		math::addScalarToTensor(Context(), weight, 0.001f);
//
//		Tensor correct(output.shape(), "float32", Device::cpu());
//		baseline_inference(input, correct, weight, bias, NonlinearityType::TANH);
//
//		math::batchnormInference(Context(), input, output, weight, bias, NonlinearityType::TANH);
//		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			Context context(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//			weight.moveTo(Device::cuda(0));
//			bias.moveTo(Device::cuda(0));
//
//			math::batchnormInference(context, input, output, weight, bias, NonlinearityType::TANH);
//			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//		}
//	}
//	TEST(TestBatchNorm, backward)
//	{
//		// storage + 0 * filters : avg (load)
//		// storage + 1 * filters : sqrt(var + epsilon) (load)
//		// storage + 2 * filters : gamma (load)
//		// storage + 3 * filters : d_sigma (tmp)
//		// storage + 4 * filters : d_beta (tmp)
//
//		Tensor input( { 123, 34 }, "float32", Device::cpu());
//		Tensor output( { 123, 34 }, "float32", Device::cpu());
//		Tensor gradient_prev( { 123, 34 }, "float32", Device::cpu());
//		Tensor gradient_next( { 123, 34 }, "float32", Device::cpu());
//
//		Tensor weight( { 2, 34 }, "float32", Device::cpu());
//		Tensor bias( { 2, 34 }, "float32", Device::cpu());
//		Tensor average( { 34 }, "float32", Device::cpu());
//		Tensor stddev( { 34 }, "float32", Device::cpu());
//
//		Tensor weight_update( { 2, 34 }, "float32", Device::cpu());
//		Tensor bias_update( { 2, 34 }, "float32", Device::cpu());
//
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(gradient_next, 1.57f);
//		testing::initForTest(weight, 0.0f);
//		testing::initForTest(bias, 1.57f);
//		math::addScalarToTensor(Context(), weight, 0.001f);
//
//		testing::initForTest(weight_update, 0.1f);
//		testing::initForTest(bias_update, 0.1f);
//
//		Tensor correct_prev(output.shape(), "float32", Device::cpu());
//		Tensor correct_weight_update( { 2, 34 }, "float32", Device::cpu());
//		Tensor correct_bias_update( { 2, 34 }, "float32", Device::cpu());
//		correct_weight_update.copyFrom(Context(), weight_update);
//		correct_bias_update.copyFrom(Context(), bias_update);
//
//		baseline_forward(input, output, weight, bias, average, stddev, NonlinearityType::SIGMOID);
//		baseline_backward(input, output, correct_prev, gradient_next, weight, average, stddev, NonlinearityType::SIGMOID);
//		baseline_update(input, gradient_next, average, stddev, correct_weight_update, correct_bias_update);
//
//		Tensor storage( { 5 + 8, 34 }, "float32", Device::cpu());
//		ml::memcpy(storage.data<float>(), Device::cpu(), average.data<float>(), Device::cpu(), sizeof(float) * 34); // copy avg to storage
//		ml::memcpy(storage.data<float>() + 34, Device::cpu(), stddev.data<float>(), Device::cpu(), sizeof(float) * 34); // copy stddev to storage
//		ml::memcpy(storage.data<float>() + 2 * 34, Device::cpu(), weight.data<float>() + 34, Device::cpu(), sizeof(float) * 34); // copy gamma to storage
//		testing::initForTest(gradient_next, 1.57f);
//		math::batchnormBackward(Context(), input, output, gradient_prev, gradient_next, weight_update, bias_update, storage,
//				NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//		EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			storage.zeroall(Context());
//			gradient_prev.zeroall(Context());
//			Context context(Device::cuda(0));
//			testing::initForTest(weight_update, 0.1f);
//			testing::initForTest(bias_update, 0.1f);
//			testing::initForTest(gradient_next, 1.57f);
//
//			ml::memcpy(storage.data<float>(), Device::cpu(), average.data<float>(), Device::cpu(), sizeof(float) * 34); // copy avg to storage
//			ml::memcpy(storage.data<float>() + 34, Device::cpu(), stddev.data<float>(), Device::cpu(), sizeof(float) * 34); // copy stddev to storage
//			ml::memcpy(storage.data<float>() + 2 * 34, Device::cpu(), weight.data<float>() + 34, Device::cpu(), sizeof(float) * 34); // copy gamma to storage
//			storage.moveTo(Device::cuda(0));
//			weight_update.moveTo(Device::cuda(0));
//			bias_update.moveTo(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//			gradient_prev.moveTo(Device::cuda(0));
//			gradient_next.moveTo(Device::cuda(0));
//			weight.moveTo(Device::cuda(0));
//			bias.moveTo(Device::cuda(0));
//			average.moveTo(Device::cuda(0));
//			stddev.moveTo(Device::cuda(0));
//
//			batchnormBackward(context, input, output, gradient_prev, gradient_next, weight_update, bias_update, storage, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
//			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//			EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-4f);
//		}
//	}
//	TEST(TestBatchNorm, learn)
//	{
//		Tensor running_stat( { 64, 34 }, "float32", Device::cpu());
//		Tensor summed_stat( { 34 }, "float32", Device::cpu());
//		Tensor correct_summed_stat( { 34 }, "float32", Device::cpu());
//		testing::initForTest(running_stat, 0.0);
//
//		baseline_learn(correct_summed_stat, running_stat, 50);
//
//		batchnormUpdate(Context(), summed_stat, running_stat, 50);
//		EXPECT_LE(testing::diffForTest(correct_summed_stat, summed_stat), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices())
//		{
//			Context context(Device::cuda(0));
//			summed_stat.moveTo(Device::cuda(0));
//			running_stat.moveTo(Device::cuda(0));
//			summed_stat.zeroall(context);
//
//			batchnormUpdate(context, summed_stat, running_stat, 50);
//			EXPECT_LE(testing::diffForTest(correct_summed_stat, summed_stat), 1.0e-4f);
//		}
//	}
//
//} /* namespace ml */

