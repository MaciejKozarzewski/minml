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

	void baseline_bn_forward(const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias, Tensor &stats, int stat_id,
			ActivationType act, float epsilon = 1.0e-6)
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
				const float gamma = weights.get( { f });
				const float beta = bias.get( { f });
				const float tmp = (input.get( { b, f }) - avg) / stddev;
				output.set(tmp * gamma + beta, { b, f });
			}
		activationForward(Context(), 1.0f, output, 0.0f, output, act);
	}
	void baseline_bn_inference(const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias, const Tensor &stats,
			ActivationType act, float epsilon = 1.0e-6)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const float avg = stats.get( { 0, j });
				const float var = stats.get( { 1, j });
				const float gamma = weights.get( { j });
				const float beta = bias.get( { j });
				const float tmp = gamma * (input.get( { i, j }) - avg) / std::sqrt(epsilon + var) + beta;
				output.at( { i, j }) = tmp;
			}
		activationForward(Context(), 1.0f, output, 0.0f, output, act);
	}
	void baseline_bn_backward(const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			const Tensor &stats, int stat_id, ActivationType act, float epsilon = 1.0e-6f)
	{
		assert(input.device().isCPU());
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.shape().lastDim();
		Tensor d_sigma( { last_dim }, DataType::FLOAT32, Device::cpu());
		Tensor d_mu( { last_dim }, DataType::FLOAT32, Device::cpu());

//		activationBackward(Context(), 1.0f, gradient_next, output, 0.0f, gradient_next, act);
		for (int b = 0; b < first_dim; b++) //apply variance, beta and gamma to output
			for (int f = 0; f < last_dim; f++)
			{
				const float gamma = 1.0e-16f + weights.get( { f });
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
				const float gamma = 1.0e-8f + weights.get( { f });
				const float avg = stats.get( { stat_id, 3 * f + 1 });
				const float var = sqrt(epsilon + stats.get( { stat_id, 3 * f + 2 }) / (first_dim - 1));
				const float in = (input.get( { b, f }) - avg) / var;
				const float m = first_dim;
				const float tmp1 = gamma * gradient_next.get( { b, f }) / var;
				const float tmp2 = d_sigma.get( { f }) * in / (m - 1);
				const float tmp3 = d_mu.get( { f }) / m;
				gradient_prev.set(tmp1 + tmp2 + tmp3, { b, f });
			}
	}
	void baseline_bn_update(const Tensor &input, const Tensor &gradient_next, const Tensor &stats, int stat_id, Tensor &weight_update,
			Tensor &bias_update, float epsilon = 1.0e-6f)
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
				d_gamma.at( { f }) = d_gamma.get( { f }) + gamma_update;
				d_beta.at( { f }) = d_beta.get( { f }) + beta_update;
			}
		for (int f = 0; f < last_dim; f++)
		{
			weight_update.at( { f }) = weight_update.get( { f }) + d_gamma.get( { f });
			bias_update.at( { f }) = bias_update.get( { f }) + d_beta.get( { f });
		}
	}
	void baseline_bn_learn(Tensor &stat, const Tensor &running_stat, int first_dim)
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

	void add_scalar_to_tensor(Tensor &tensor, double scalar)
	{
		assert(tensor.device().isCPU());
		Tensor tmp = tensor.view( { tensor.volume() });
		for (int i = 0; i < tmp.volume(); i++)
			tmp.at( { i }) = (float) tmp.at( { i }) + scalar;
	}

	template<typename T>
	void baseline_ln_forward(const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias, const Tensor &ext, T epsilon = 0.0e-6)
	{
		assert(input.device().isCPU());
		assert(input.rank() == 2);
		const int first_dim = input.shape().firstDim();
		const int last_dim = input.shape().lastDim();

		Tensor tmp_in(input.shape(), input.dtype(), input.device());
		if (not ext.isEmpty())
			addTensors(Context(), tmp_in, input, ext);
		else
			tmp_in = input;

		for (int i = 0; i < first_dim; i++)
		{
			T avg = 0;
			for (int j = 0; j < last_dim; j++)
				avg += static_cast<T>(tmp_in.at( { i, j }));
			avg /= last_dim;

			T var = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T tmp = static_cast<T>(tmp_in.at( { i, j })) - avg;
				var += tmp * tmp;
			}

			const T stddev = std::sqrt(epsilon + var / (last_dim - 1));

			for (int j = 0; j < last_dim; j++)
			{
				const T gamma = weight.at( { j });
				const T beta = bias.at( { j });
				const T tmp = gamma * (static_cast<T>(tmp_in.at( { i, j })) - avg) / stddev + beta;
				output.at( { i, j }) = tmp;
			}
		}
	}
	template<typename T>
	void baseline_ln_backward(const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weight, Tensor &weight_update,
			Tensor &bias_update, T epsilon = 0.0e-6)
	{
		assert(input.device().isCPU());
		assert(input.rank() == 2);
		const int first_dim = input.shape().firstDim();
		const int last_dim = input.shape().lastDim();

		for (int i = 0; i < first_dim; i++)
		{
//			T mu = 0;
//			for (int j = 0; j < last_dim; j++)
//				mu += static_cast<T>(input.at( { i, j }));
//			mu /= last_dim;
//
//			T sq = 0;
//			for (int j = 0; j < last_dim; j++)
//			{
//				const T tmp = static_cast<T>(input.at( { i, j })) - mu;
//				sq += tmp * tmp;
//			}
//			const T var = sq / (last_dim - 1);
//			const T sqrtvar = std::sqrt(var + epsilon);
//
//			T d_sigma = 0;
//			T d_mu = 0;
//			for (int j = 0; j < last_dim; j++)
//			{
//				const T in = input.at( { i, j });
//				const T grad = gradient_next.at( { i, j });
//				const T gamma = weight.at( { j });
//
//				d_sigma -= grad * gamma * (in - mu);
//				d_mu -= grad * gamma;
//			}
//			d_sigma *= 0.5 / (sqrtvar * sqrtvar * sqrtvar * (last_dim - 1));
//			d_mu *= static_cast<T>(1) / (sqrtvar * last_dim);
//
//			for (int j = 0; j < last_dim; j++)
//			{
//				const T in = input.at( { i, j });
//				const T grad = gradient_next.at( { i, j });
//				const T gamma = weight.at( { j });
//
//				const T dy = grad * gamma / sqrtvar + d_sigma * 2 * (in - mu) + d_mu;
//				gradient_prev.at( { i, j }) = dy;
//			}

			T avg = 0;
			for (int j = 0; j < last_dim; j++)
				avg += static_cast<T>(input.at( { i, j }));
			avg /= last_dim;

			T var = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T tmp = static_cast<T>(input.at( { i, j })) - avg;
				var += tmp * tmp;
			}
			const T stddev = std::sqrt(epsilon + var / (last_dim - 1));

			T d_sigma = 0;
			T d_mu = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T in = input.at( { i, j });
				const T grad = gradient_next.at( { i, j });
				const T gamma = weight.at( { j });
				const T tmp = (in - avg) / stddev;
				d_sigma -= gamma * grad * tmp / stddev;
				d_mu -= gamma * grad / stddev;
			}

			for (int j = 0; j < last_dim; j++)
			{
				const T in = input.at( { i, j });
				const T grad = gradient_next.at( { i, j });
				const T gamma = weight.at( { j });
				const T tmp1 = gamma * grad / stddev;
				const T tmp2 = d_sigma * (in - avg) / stddev / (last_dim - 1);
				const T tmp3 = d_mu / last_dim;
				gradient_prev.at( { i, j }) = (tmp1 + tmp2 + tmp3);
			}

			for (int j = 0; j < last_dim; j++)
			{
				const T in = input.at( { i, j });
				const T grad = gradient_next.at( { i, j });
				const T gamma_update = grad * (in - avg) / stddev;
				const T beta_update = grad;

				const T wu = weight_update.at( { j });
				const T bu = bias_update.at( { j });
				weight_update.at( { j }) = (wu + gamma_update);
				bias_update.at( { j }) = (bu + beta_update);
			}
		}
	}

	template<typename T>
	void baseline_rmsnorm_forward(const Tensor &input, Tensor &output, const Tensor &weights, T epsilon = 0.0e-6)
	{
		const bool use_gamma = not weights.isEmpty();
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.lastDim();

		for (int i = 0; i < first_dim; i++)
		{
			T sum_square = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T in = input.at( { i, j });
				sum_square += in * in;
			}
			const T rms = std::sqrt(sum_square / last_dim);

			const T inv_rms = static_cast<T>(1) / (epsilon + rms);
			for (int j = 0; j < last_dim; j++)
			{
				const T gamma = use_gamma ? weights.at( { j }) : 1.0f;
				const T in = input.at( { i, j });
				output.at( { i, j }) = (gamma * in * inv_rms);
			}
		}
	}
	template<typename T>
	void baseline_rmsnorm_backward(const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights, Tensor &weights_update,
			T epsilon = 0.0e-6)
	{
		const bool use_gamma = not weights.isEmpty();
		const int first_dim = input.shape().volumeWithoutLastDim();
		const int last_dim = input.lastDim();

		for (int i = 0; i < first_dim; i++)
		{
			T sum_square = 0, sum = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T in = input.at( { i, j });
				const T grad = gradient_next.at( { i, j });
				const T gamma = use_gamma ? weights.at( { j }) : 1.0f;
				sum_square += in * in;
				sum += in * grad * gamma;
			}
			const T rms = std::sqrt(sum_square / last_dim);

			const T inv_rms = static_cast<T>(1) / (epsilon + rms);
			for (int j = 0; j < last_dim; j++)
			{
				const T gamma = use_gamma ? weights.at( { j }) : 1.0f;
				const T in = input.at( { i, j });
				const T out = in * inv_rms;
				const T grad = gradient_next.at( { i, j });

				if (use_gamma)
				{
					const T wu = weights_update.at( { j });
					weights_update.at( { j }) = wu + grad * out;
				}
				gradient_prev.at( { i, j }) = (gamma * grad * sum_square - in * sum) / (last_dim * rms * rms * rms);
			}
		}
	}

	class BaselineLN: public Layer
	{
		public:
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return getInputShape();
			}
			Shape getWeightShape() const
			{
				return Shape( { getInputShape().lastDim() });
			}
			Shape getBiasShape() const
			{
				return Shape( { getInputShape().lastDim() });
			}
			std::string name() const
			{
				return "BaselineLN";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				return std::make_unique<BaselineLN>();
			}
			void init()
			{
				ml::testing::initRandom(getWeights().getParam());
				ml::testing::initRandom(getBias().getParam());
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_ln_forward<float>(input[0], output, getWeights().getParam(), getBias().getParam(), Tensor());
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_ln_forward<double>(input[0], output, getWeights().getParam(), getBias().getParam(), Tensor());
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_ln_backward<float>(input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient(),
							getBias().getGradient());
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_ln_backward<double>(input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient(),
							getBias().getGradient());
			}
	};
	class BaselineRMSN: public Layer
	{
		public:
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return getInputShape();
			}
			Shape getWeightShape() const
			{
				return Shape( { getInputShape().lastDim() });
			}
			std::string name() const
			{
				return "BaselineRMSN";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				return std::make_unique<BaselineRMSN>();
			}
			void init()
			{
				ml::testing::initRandom(getWeights().getParam());
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_rmsnorm_forward<float>(input[0], output, getWeights().getParam());
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_rmsnorm_forward<double>(input[0], output, getWeights().getParam());
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_rmsnorm_backward<float>(input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient());
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_rmsnorm_backward<double>(input[0], gradient_prev[0], gradient_next, getWeights().getParam(), getWeights().getGradient());
			}
	};
}

namespace ml
{
	TEST(TestBatchNorm, forward)
	{
		const int batch_size = 1;
		const int height = 2;
		const int width = 3;
		const int filters = 12;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float32", Device::cpu());
		std::vector<Tensor> output;
		for (int i = 0; i < 10; i++)
			output.push_back(Tensor(input.shape(), "float32", Device::cpu()));

		Tensor weights( { filters }, "float32", Device::cpu());
		Tensor bias = zeros_like(weights);
		Tensor correct_stats( { (int) output.size(), 3 * filters }, "float32", Device::cpu());

		testing::initForTest(weights, 0.0f);
		add_scalar_to_tensor(weights, 1.001f);
		testing::initForTest(bias, 0.0f);

		std::vector<Tensor> correct;
		for (size_t i = 0; i < output.size(); i++)
			correct.push_back(Tensor(input.shape(), "float32", Device::cpu()));
		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		for (size_t i = 0; i < output.size(); i++)
		{
			testing::initForTest(input, 0.1f * i);
			baseline_bn_forward(input, correct[i], weights, bias, correct_stats, i, ActivationType::SIGMOID);
//			Tensor stats = running_stats.view( { running_stats.lastDim() }, i * running_stats.lastDim());
//			batchnormForward(context, 1.0f, input, weight, 0.0f, output[i], stats, ActivationType::SIGMOID);
//
//			Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
//			Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);
//			EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-4f);
//			EXPECT_LE(testing::diffForTest(correct[i], output[i]), 1.0e-4f);
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
			weights.moveTo(device);
			bias.moveTo(device);
			running_stats.moveTo(device);
			running_stats.zeroall();

			for (size_t i = 0; i < output.size(); i++)
			{
				testing::initForTest(input, 0.1f * i);
				Tensor stats = running_stats.view( { running_stats.lastDim() }, { (int) i, 0 });
				batchnormForward(context, 1.0f, input, weights, bias, 0.0f, output[i], stats, ActivationType::SIGMOID);
				context.synchronize();

				Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
				Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);

				EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-3f);
				EXPECT_LE(testing::diffForTest(correct[i], output[i]), 1.0e-4f);
			}
		}
	}
	TEST(TestBatchNorm, inference)
	{
//		Tensor input( { 123, 34 }, "float32", Device::cpu());
//		Tensor output( { 123, 34 }, "float32", Device::cpu());
//		Tensor weight( { 4, 34 }, "float32", Device::cpu());
//
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(weight, 0.0f);
//		add_scalar_to_tensor(weight, 1.1f);
//
//		Tensor correct(output.shape(), "float32", Device::cpu());
//		baseline_bn_inference(input, correct, weight, ActivationType::TANH);
//
//		batchnormInference(Context(), 1.0f, input, weight, 0.0f, output, ActivationType::TANH);
//		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			output.moveTo(device);
//			weight.moveTo(device);
//			output.zeroall();
//			context.synchronize();
//
//			batchnormInference(context, 1.0f, input, weight, 0.0f, output, ActivationType::TANH);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
//		}
	}
	TEST(TestBatchNorm, backward)
	{
		const int batch_size = 11;
		const int height = 12;
		const int width = 13;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor gradient_next(input.shape(), "float32", Device::cpu());

		Tensor weights( { filters }, "float32", Device::cpu());
		Tensor bias = zeros_like(weights);
		Tensor stats( { 10, 3 * filters }, "float32", Device::cpu());

		Tensor weights_update = zeros_like(weights);
		Tensor bias_update = zeros_like(bias);

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weights, 0.0f);
		add_scalar_to_tensor(weights, 1.1f);
		testing::initForTest(bias, 0.0f);

		Tensor correct_prev = zeros_like(gradient_prev);
		Tensor correct_weights_update = zeros_like(weights_update);
		Tensor correct_bias_update = zeros_like(bias_update);

		const int stat_id = 2;

		baseline_bn_forward(input, output, weights, bias, stats, stat_id, ActivationType::LINEAR);
		baseline_bn_backward(input, output, correct_prev, gradient_next, weights, stats, stat_id, ActivationType::LINEAR);
		baseline_bn_update(input, gradient_next, stats, stat_id, correct_weights_update, correct_bias_update);

		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		output.zeroall();
		testing::initForTest(gradient_next, 1.57f);
		Tensor stats2 = running_stats.view( { running_stats.lastDim() }, { stat_id, 0 });
//		batchnormForward(context, 1.0f, input, weight, 0.0f, output, stats2, ActivationType::SIGMOID);
//		batchnormBackward(context, 1.0f, input, output, gradient_next, weight, 0.0f, gradient_prev, 1.0f, weight_update, stats2,
//				ActivationType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(gradient_next, 1.57f);

			weights_update.moveTo(device);
			bias_update.moveTo(device);
			input.moveTo(device);
			output.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			running_stats.moveTo(device);

			output.zeroall();
			weights_update.zeroall();
			bias_update.zeroall();
			running_stats.zeroall();
			gradient_prev.zeroall();

			Tensor stats2 = running_stats.view( { running_stats.lastDim() }, { stat_id, 0 });
			batchnormForward(context, 1.0f, input, weights, bias, 0.0f, output, stats2, ActivationType::LINEAR);
			batchnormBackward(context, 1.0f, input, output, gradient_next, weights, bias, 0.0f, gradient_prev, 1.0f, weights_update, bias_update,
					stats2, ActivationType::LINEAR);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		}
	}

	TEST(TestBatchNorm, forward_fp16)
	{
		const int batch_size = 1;
		const int height = 2;
		const int width = 3;
		const int filters = 12;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float16", Device::cpu());
		std::vector<Tensor> outputs;
		for (int i = 0; i < 10; i++)
			outputs.push_back(zeros_like(input));

		Tensor weights( { filters }, "float16", Device::cpu());
		Tensor bias = zeros_like(weights);
		Tensor correct_stats( { (int) outputs.size(), 3 * filters }, "float32", Device::cpu());

		testing::initForTest(weights, 0.0f);
		add_scalar_to_tensor(weights, 1.001f);
		testing::initForTest(bias, 0.0f);

		std::vector<Tensor> correct_outputs;
		for (size_t i = 0; i < outputs.size(); i++)
			correct_outputs.push_back(zeros_like(input));
		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		for (size_t i = 0; i < outputs.size(); i++)
		{
			testing::initForTest(input, 0.1f * i);
			baseline_bn_forward(input, correct_outputs[i], weights, bias, correct_stats, i, ActivationType::SIGMOID);
//			batchnormForward(context, 1.0f, input, weight, 0.0f, output[i], running_stats, i, ActivationType::SIGMOID);
//
//			EXPECT_LE(testing::diffForTest(correct[i], output[i]), 1.0e-4f);
//			Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
//			Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);
//			EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-4f);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			for (size_t i = 0; i < outputs.size(); i++)
			{
				outputs[i].moveTo(device);
				outputs[i].zeroall();
			}
			weights.moveTo(device);
			bias.moveTo(device);
			running_stats.moveTo(device);
			running_stats.zeroall();

			for (size_t i = 0; i < outputs.size(); i++)
			{
				testing::initForTest(input, 0.1f * i);
				Tensor stats = running_stats.view( { running_stats.lastDim() }, { (int) i, 0 });
				batchnormForward(context, 1.0f, input, weights, bias, 0.0f, outputs[i], stats, ActivationType::SIGMOID);
				context.synchronize();

				Tensor stats1 = correct_stats.view( { 3 * filters }, i * 3 * filters);
				Tensor stats2 = running_stats.view( { 3 * filters }, i * 3 * filters);
				EXPECT_LE(testing::diffForTest(stats1, stats2), 1.0e-4f);
				EXPECT_LE(testing::diffForTest(correct_outputs[i], outputs[i]), 1.0e-3f);
			}
		}
	}
	TEST(TestBatchNorm, inference_fp16)
	{
//		Tensor input( { 123, 34 }, "float16", Device::cpu());
//		Tensor output = zeros_like(input);
//		Tensor weight( { 4, 34 }, "float16", Device::cpu());
//
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(weight, 0.0f);
//		add_scalar_to_tensor(weight, 1.1f);
//
//		Tensor correct_output = zeros_like(output);
//		baseline_bn_inference(input, correct_output, weight, ActivationType::TANH);
//
////		batchnormInference(Context(), 1.0f, input, weight, 0.0f, output, ActivationType::LINEAR);
////		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//
//		if (testing::has_device_supporting(DataType::FLOAT16))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			output.moveTo(device);
//			weight.moveTo(device);
//			output.zeroall();
//			context.synchronize();
//
//			batchnormInference(context, 1.0f, input, weight, 0.0f, output, ActivationType::TANH);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);
//		}
	}
	TEST(TestBatchNorm, backward_fp16)
	{
		const int batch_size = 11;
		const int height = 12;
		const int width = 13;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size * height * width, filters }, "float16", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor gradient_prev = zeros_like(input);
		Tensor gradient_next = zeros_like(input);

		Tensor weights( { filters }, "float16", Device::cpu());
		Tensor bias = zeros_like(weights);
		Tensor stats( { 10, 3 * filters }, "float32", Device::cpu());

		Tensor weights_update = zeros_like(weights);
		Tensor bias_update = zeros_like(bias);

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weights, 0.0f);
		add_scalar_to_tensor(weights, 1.1f);
		testing::initForTest(bias, 0.0f);

		Tensor correct_prev = zeros_like(gradient_prev);
		Tensor correct_weights_update = zeros_like(weights_update);
		Tensor correct_bias_update = zeros_like(bias_update);

		const int stat_id = 2;

		baseline_bn_forward(input, output, weights, bias, stats, stat_id, ActivationType::LINEAR);
		baseline_bn_backward(input, output, correct_prev, gradient_next, weights, stats, stat_id, ActivationType::LINEAR);
		baseline_bn_update(input, gradient_next, stats, stat_id, correct_weights_update, correct_bias_update);

		Tensor running_stats( { 64, 3 * filters }, "float32", Device::cpu());
		output.zeroall();
		testing::initForTest(gradient_next, 1.57f);
		Tensor stats2 = running_stats.view( { running_stats.lastDim() }, { stat_id, 0 });
		//		batchnormForward(context, 1.0f, input, weight, 0.0f, output, stats2, ActivationType::SIGMOID);
		//		batchnormBackward(context, 1.0f, input, output, gradient_next, weight, 0.0f, gradient_prev, 1.0f, weight_update, stats2,
		//				ActivationType::SIGMOID);
		//		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(gradient_next, 1.57f);

			weights_update.moveTo(device);
			bias_update.moveTo(device);
			input.moveTo(device);
			output.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			running_stats.moveTo(device);

			output.zeroall();
			weights_update.zeroall();
			bias_update.zeroall();
			running_stats.zeroall();
			gradient_prev.zeroall();

			Tensor stats2 = running_stats.view( { running_stats.lastDim() }, { stat_id, 0 });
			batchnormForward(context, 1.0f, input, weights, bias, 0.0f, output, stats2, ActivationType::LINEAR);
			batchnormBackward(context, 1.0f, input, output, gradient_next, weights, bias, 0.0f, gradient_prev, 1.0f, weights_update, bias_update,
					stats2, ActivationType::LINEAR);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-2f);
			EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-3f);
			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-3f);
		}
	}
//	TEST(TestBatchNorm, learn)
//	{
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
//	}

//	TEST(TestLayerNorm, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineLN() };
//		gradcheck.setInputShape(Shape( { 1, 3 }));
//
//		gradcheck.check(3, 1.0e-4, "all");
//
//		exit(0);
//	}
	TEST(TestLayerNorm, forward)
	{
		const int batch_size = 123;
		const int filters = 44;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());

		Tensor weight( { filters }, "float32", Device::cpu());
		Tensor bias( { filters }, "float32", Device::cpu());

		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.1f);
		testing::initForTest(bias, 1.0f);

		Tensor correct(input.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		baseline_ln_forward<float>(input, correct, weight, bias, Tensor());
		layernormForward(context, input, output, weight, bias, Tensor());

		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			weight.moveTo(device);
			bias.moveTo(device);

			testing::initForTest(input, 0.0f);
			layernormForward(context, input, output, weight, bias, Tensor());
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
		}
	}
	TEST(TestLayerNorm, backward)
	{
		const int batch_size = 123;
		const int filters = 44;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor gradient_next(input.shape(), "float32", Device::cpu());

		Tensor weight( { filters }, "float32", Device::cpu());

		Tensor weight_update( { filters }, "float32", Device::cpu());
		Tensor bias_update( { filters }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weight, 0.1f);
		add_scalar_to_tensor(weight, 1.1f);

		testing::initForTest(weight_update, 0.1f);
		testing::initForTest(bias_update, 0.2f);

		Tensor correct_prev(input.shape(), "float32", Device::cpu());
		Tensor correct_weight_update( { filters }, "float32", Device::cpu());
		correct_weight_update.copyFrom(Context(), weight_update);
		Tensor correct_bias_update( { filters }, "float32", Device::cpu());
		correct_bias_update.copyFrom(Context(), bias_update);

		baseline_ln_backward<float>(input, correct_prev, gradient_next, weight, correct_weight_update, correct_bias_update);

		testing::initForTest(gradient_next, 1.57f);
		layernormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, bias_update, 0.0f);
		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 0.1f);
			testing::initForTest(bias_update, 0.2f);
			testing::initForTest(gradient_next, 1.57f);

			input.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weight.moveTo(device);

			weight_update.moveTo(device);
			bias_update.moveTo(device);

			gradient_prev.zeroall();

			layernormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, bias_update, 0.0f);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_bias_update, bias_update), 1.0e-4f);
		}
	}

//	TEST(TestRMSNorm, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineRMSN() };
//		gradcheck.setInputShape(Shape( { 31, 43 }));
//
//		gradcheck.check(1000, 1.0e-4, "all");
//
//		exit(0);
//	}
	TEST(TestRMSNorm, forward)
	{
		const int batch_size = 123;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());

		Tensor weight( { filters }, "float32", Device::cpu());

		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.1f);

		Tensor correct(input.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		baseline_rmsnorm_forward<float>(input, correct, weight, 1.0e-6f);

		rmsnormForward(context, input, output, weight);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			weight.moveTo(device);

			testing::initForTest(input, 0.0f);
			rmsnormForward(context, input, output, weight);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
		}
	}
	TEST(TestRMSNorm, backward)
	{
		const int batch_size = 123;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor gradient_next(input.shape(), "float32", Device::cpu());

		Tensor weight( { filters }, "float32", Device::cpu());

		Tensor weight_update( { filters }, "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);
		testing::initForTest(weight, 0.0f);
		add_scalar_to_tensor(weight, 1.1f);

		testing::initForTest(weight_update, 0.1f);

		Tensor correct_prev(input.shape(), "float32", Device::cpu());
		Tensor correct_weight_update( { filters }, "float32", Device::cpu());
		correct_weight_update.copyFrom(Context(), weight_update);

		baseline_rmsnorm_backward<float>(input, correct_prev, gradient_next, weight, correct_weight_update, 1.0e-6f);

		testing::initForTest(gradient_next, 1.57f);
		rmsnormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, 0.0f);
		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 0.1f);
			testing::initForTest(gradient_next, 1.57f);

			input.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weight.moveTo(device);

			weight_update.moveTo(device);

			gradient_prev.zeroall();

			rmsnormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, 0.0f);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

	TEST(TestRMSNorm, forward_no_gamma)
	{
		const int batch_size = 123;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());
		Tensor weight;

		Tensor correct(input.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0f);
		baseline_rmsnorm_forward<float>(input, correct, weight, 1.0e-6f);

		rmsnormForward(context, input, output, weight);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			weight.moveTo(device);

			testing::initForTest(input, 0.0f);
			rmsnormForward(context, input, output, weight);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
		}
	}
	TEST(TestRMSNorm, backward_no_gamma)
	{
		const int batch_size = 123;
		const int filters = 43;
		Context context;

		Tensor input( { batch_size, filters }, "float32", Device::cpu());
		Tensor gradient_prev(input.shape(), "float32", Device::cpu());
		Tensor gradient_next(input.shape(), "float32", Device::cpu());

		Tensor weight, weight_update;

		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.57f);

		Tensor correct_prev(input.shape(), "float32", Device::cpu());
		Tensor correct_weight_update( { filters }, "float32", Device::cpu());
		correct_weight_update.copyFrom(Context(), weight_update);

		baseline_rmsnorm_backward<float>(input, correct_prev, gradient_next, weight, correct_weight_update, 1.0e-6f);

		testing::initForTest(gradient_next, 1.57f);
		rmsnormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, 0.0f);
		EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(gradient_next, 1.57f);

			input.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weight.moveTo(device);

			weight_update.moveTo(device);

			gradient_prev.zeroall();

			rmsnormBackward(context, input, gradient_prev, gradient_next, weight, weight_update, 0.0f);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_prev, gradient_prev), 1.0e-4f);
		}
	}

} /* namespace ml */

