/*
 * test_global_pooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/LearnableGlobalPooling.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/json.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;
	template<typename T = float>
	void baseline_pooling_forward(const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 4);
		assert(output.rank() == 2);
		assert(input.firstDim() == output.firstDim());
		assert(2 * input.lastDim() == output.lastDim());

		const int channels = input.lastDim();
		const T inv = 1.0 / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
			{
				T avg_value = 0.0;
				T max_value = input.at( { i, 0, 0, l });
				for (int j = 0; j < input.dim(1); j++)
					for (int k = 0; k < input.dim(2); k++)
					{
						const T x = input.get( { i, j, k, l });
						avg_value += x;
						max_value = std::max(max_value, x);
					}
				output.at( { i, 0 * channels + l }) = avg_value * inv;
				output.at( { i, 1 * channels + l }) = max_value;
			}
	}
	template<typename T = float>
	void baseline_pooling_backward(Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input, const Tensor &output)
	{
		assert(input.shape() == gradient_prev.shape());
		assert(output.shape() == gradient_next.shape());
		assert(input.rank() == 4);
		assert(output.rank() == 2);
		assert(input.firstDim() == output.firstDim());
		assert(2 * input.lastDim() == output.lastDim());

		const int channels = input.lastDim();
		const T inv = 1.0 / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.dim(0); i++)
			for (int l = 0; l < input.dim(3); l++)
			{
				int max_idx_h = 0;
				int max_idx_w = 0;
				T max_value = input.at( { i, 0, 0, l });
				for (int j = 0; j < input.dim(1); j++)
					for (int k = 0; k < input.dim(2); k++)
					{
						const T in = input.at( { i, j, k, l });
						const T grad_avg = gradient_next.at( { i, 0 * channels + l });
						gradient_prev.at( { i, j, k, l }) = (T) gradient_prev.at( { i, j, k, l }) + inv * grad_avg;
						if (in > max_value)
						{
							max_value = in;
							max_idx_h = j;
							max_idx_w = k;
						}
					}

				const T grad_max = gradient_next.at( { i, 1 * channels + l });
				gradient_prev.at( { i, max_idx_h, max_idx_w, l }) = (T) gradient_prev.at( { i, max_idx_h, max_idx_w, l }) + grad_max;
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
//		activationForward(Context(), output, output, act);
	}
	void baseline_broadcasting_backward(Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, ActivationType act)
	{
		assert(output.rank() == 4);
		assert(gradient_next.shape() == output.shape());
		assert(gradient_prev.rank() == 2);
		assert(gradient_prev.firstDim() == output.firstDim());
		assert(gradient_prev.lastDim() == output.lastDim());

//		activationBackward(Context(), gradient_next, gradient_next, output, act, 0.0f);
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

	template<typename T = float>
	void baseline_average_pooling_forward(const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 4);
		assert(output.rank() == 2);
		assert(input.firstDim() == output.firstDim());
		assert(input.lastDim() == output.lastDim());

		const T inv = 1.0 / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
			{
				T avg_value = 0.0;
				for (int j = 0; j < input.dim(1); j++)
					for (int k = 0; k < input.dim(2); k++)
						avg_value += (T) input.at( { i, j, k, l });
				output.at( { i, l }) = avg_value * inv;
			}
	}
	template<typename T = float>
	void baseline_average_pooling_backward(Tensor &gradient_prev, const Tensor &gradient_next)
	{
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 2);
		assert(gradient_prev.firstDim() == gradient_next.firstDim());
		assert(gradient_prev.lastDim() == gradient_next.lastDim());

		const T inv = 1.0 / (gradient_prev.dim(1) * gradient_prev.dim(2));
		for (int i = 0; i < gradient_prev.dim(0); i++)
			for (int l = 0; l < gradient_prev.dim(3); l++)
				for (int j = 0; j < gradient_prev.dim(1); j++)
					for (int k = 0; k < gradient_prev.dim(2); k++)
						gradient_prev.at( { i, j, k, l }) = inv * (T) gradient_next.at( { i, l });
	}

	template<typename T = float>
	void baseline_learnable_pooling_forward(const Tensor &input, const Tensor &weights, Tensor &output)
	{
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
				for (int out = 0; out < weights.firstDim(); out++)
				{
					T acc = 0.0;
					int m = 0;
					for (int j = 0; j < input.dim(1); j++)
						for (int k = 0; k < input.dim(2); k++, m++)
							acc += (T) input.at( { i, j, k, l }) * (T) weights.at( { out, m });
					output.at( { i, l * weights.firstDim() + out }) = acc;
				}
	}
	template<typename T = float>
	void baseline_learnable_pooling_backward(Tensor &gradient_prev, const Tensor &weights, const Tensor &gradient_next)
	{
		gradient_prev.zeroall();
		for (int i = 0; i < gradient_prev.firstDim(); i++)
			for (int l = 0; l < gradient_prev.lastDim(); l++)
				for (int out = 0; out < weights.firstDim(); out++)
				{
					const T grad = (T) gradient_next.at( { i, l * weights.firstDim() + out });
					int m = 0;
					for (int j = 0; j < gradient_prev.dim(1); j++)
						for (int k = 0; k < gradient_prev.dim(2); k++, m++)
							gradient_prev.at( { i, j, k, l }) = (T) gradient_prev.at( { i, j, k, l }) + grad * (T) weights.at( { out, m });
				}
	}
	template<typename T = float>
	void baseline_learnable_pooling_update(const Tensor &input, Tensor &weights_update, const Tensor &gradient_next)
	{
		weights_update.zeroall();
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
				for (int out = 0; out < weights_update.firstDim(); out++)
				{
					const T grad = (T) gradient_next.at( { i, l * weights_update.firstDim() + out });
					int m = 0;
					for (int j = 0; j < input.dim(1); j++)
						for (int k = 0; k < input.dim(2); k++, m++)
							weights_update.at( { out, m }) = (T) weights_update.at( { out, m }) + grad * (T) input.at( { i, j, k, l });
				}
	}

	template<typename T = float>
	void baseline_channel_scaling_forward(const Tensor &input, Tensor &output, const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(input.shape() == output.shape());
		assert(scales.rank() == 2);
		assert(scales.firstDim() == output.firstDim());
		assert(scales.lastDim() == output.lastDim());

		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
						output.at( { i, j, k, l }) = (T) input.at( { i, j, k, l }) * (T) scales.at( { i, l });
	}
	template<typename T = float>
	void baseline_channel_scaling_backward(Tensor &gradient_prev, Tensor &gradient_scales, const Tensor &gradient_next, const Tensor &input,
			const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(gradient_prev.shape() == input.shape());
		assert(gradient_scales.shape() == scales.shape());
		assert(gradient_prev.shape() == gradient_next.shape());

		for (int i = 0; i < gradient_next.dim(0); i++)
			for (int l = 0; l < gradient_next.dim(3); l++)
			{
				T grad = 0;
				for (int j = 0; j < gradient_next.dim(1); j++)
					for (int k = 0; k < gradient_next.dim(2); k++)
					{
						grad += (T) gradient_next.at( { i, j, k, l }) * (T) input.at( { i, j, k, l });
						gradient_prev.at( { i, j, k, l }) = (T) scales.at( { i, l }) * (T) gradient_next.at( { i, j, k, l });
					}
				gradient_scales.at( { i, l }) = grad;
			}
	}

	class BaselineGlobalPooling: public Layer
	{
		public:
			BaselineGlobalPooling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int channels = getInputShape().lastDim();
				return Shape( { batch_size, 2 * channels });
			}
			std::string name() const
			{
				return "BaselineGlobalPooling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineGlobalPooling> result = std::make_unique<BaselineGlobalPooling>();
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_pooling_forward<float>(input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_pooling_forward<double>(input[0], output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_pooling_backward<float>(gradient_prev[0], gradient_next, input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_pooling_backward<double>(gradient_prev[0], gradient_next, input[0], output);
			}
	};
	class BaselineGlobalAveragePooling: public Layer
	{
		public:
			BaselineGlobalAveragePooling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int channels = getInputShape().lastDim();
				return Shape( { batch_size, channels });
			}
			std::string name() const
			{
				return "BaselineGlobalAveragePooling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineGlobalAveragePooling> result = std::make_unique<BaselineGlobalAveragePooling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_average_pooling_forward<float>(input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_average_pooling_forward<double>(input[0], output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_average_pooling_backward<float>(gradient_prev[0], gradient_next);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_average_pooling_backward<double>(gradient_prev[0], gradient_next);
			}
	};
	class BaselineLearbableGlobalPooling: public Layer
	{
			int m_dim = 0;
		public:
			BaselineLearbableGlobalPooling(int dim) :
					Layer("linear"),
					m_dim(dim)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int channels = getInputShape().lastDim();
				return Shape( { batch_size, m_dim * channels });
			}
			Shape getWeightShape() const
			{
				return Shape( { m_dim, getInputShape().dim(1) * getInputShape().dim(2) });
			}
			std::string name() const
			{
				return "BaselineLearbableGlobalPooling";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["dim"] = m_dim;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineLearbableGlobalPooling> result = std::make_unique<BaselineLearbableGlobalPooling>(config["dim"].getInt());
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_learnable_pooling_forward<float>(input[0], getWeights().getParam(), output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_learnable_pooling_forward<double>(input[0], getWeights().getParam(), output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
				{
					baseline_learnable_pooling_backward<float>(gradient_prev[0], getWeights().getParam(), gradient_next);
					baseline_learnable_pooling_update<float>(input[0], getWeights().getGradient(), gradient_next);
				}
				if (input[0].dtype() == DataType::FLOAT64)
				{
					baseline_learnable_pooling_backward<double>(gradient_prev[0], getWeights().getParam(), gradient_next);
					baseline_learnable_pooling_update<double>(input[0], getWeights().getGradient(), gradient_next);
				}
			}
	};
	class BaselineChannelScaling: public Layer
	{
		public:
			BaselineChannelScaling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return getInputShape(0);
			}
			std::string name() const
			{
				return "BaselineChannelScaling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineChannelScaling> result = std::make_unique<BaselineChannelScaling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_scaling_forward<float>(input[0], output, input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_scaling_forward<double>(input[0], output, input[1]);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_scaling_backward<float>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_scaling_backward<double>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
			}
	};

}

namespace ml
{
//	TEST(TestGlobalPooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineGlobalPooling() };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(1000, 1.0e-4, "all");
//
//		exit(0);
//	}
//	TEST(TestGlobalAveragePooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineGlobalAveragePooling() };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestLearnableGlobalPooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineLearbableGlobalPooling(1) };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestChannelScaling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineChannelScaling() };
//		std::vector<Shape> shapes = { Shape( { 13, 7, 8, 34 }), Shape( { 13, 34 }) };
//		gradcheck.setInputShape(shapes);
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestGlobalPooling, forward_fp32)
//	{
//		Context context;
//		Tensor input( { 12, 13, 14, 134 }, "float32", Device::cpu());
//		Tensor output( { input.firstDim(), 2 * input.lastDim() }, input.dtype(), Device::cpu());
//		Tensor correct_output = zeros_like(output);
//
//		testing::initForTest(input, 0.0);
//
//		baseline_pooling_forward(input, correct_output);
//		globalAvgAndMaxPoolingForward(context, input, output);
//
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			testing::initForTest(output, 1.57f);
//			input.moveTo(device);
//			output.moveTo(device);
//			output.zeroall();
//			globalAvgAndMaxPoolingForward(context, input, output);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
//		}
//	}
//	TEST(TestGlobalPooling, backward)
//	{
//		Context context;
//		Tensor input( { 12, 13, 14, 34 }, "float32", Device::cpu());
//		Tensor gradient_prev = zeros_like(input);
//		Tensor output( { input.firstDim(), 2 * input.lastDim() }, "float32", Device::cpu());
//		Tensor gradient_next = zeros_like(output);
//
//		Tensor correct_gradient_prev = zeros_like(gradient_prev);
//
//		testing::initForTest(input, 0.0);
//		testing::initForTest(gradient_next, 1.57);
//
//		baseline_pooling_forward(input, output);
//		baseline_pooling_backward(correct_gradient_prev, gradient_next, input, output);
//
//		output.zeroall();
//		globalAvgAndMaxPoolingForward(context, input, output);
//		globalAvgAndMaxPoolingBackward(context, gradient_prev, gradient_next, input, output, 0.0f);
//
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			gradient_prev.moveTo(device);
//			output.moveTo(device);
//			gradient_next.moveTo(device);
//			gradient_prev.zeroall();
//			output.zeroall();
//			globalAvgAndMaxPoolingForward(context, input, output);
//			globalAvgAndMaxPoolingBackward(context, gradient_prev, gradient_next, input, output, 0.0f);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//		}
//	}

	TEST(TestGlobalAveragePooling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor output( { input.firstDim(), input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_average_pooling_forward(input, correct_output);

//		globalAveragePoolingForward(context, input, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestGlobalAveragePooling, backward_fp32)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.lastDim() }, "float32", Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_average_pooling_backward(correct_gradient_prev, gradient_next);

//		globalAveragePoolingBackward(context, gradient_prev, gradient_next);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			globalAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6);
		}
	}
	TEST(TestGlobalAveragePooling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor output( { input.firstDim(), input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_average_pooling_forward(input, correct_output);

//		globalAveragePoolingForward(context, input, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4);
		}
	}
	TEST(TestGlobalAveragePooling, backward_fp16)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.lastDim() }, gradient_prev.dtype(), Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_average_pooling_backward(correct_gradient_prev, gradient_next);

//		globalAveragePoolingBackward(context, gradient_prev, gradient_next);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			globalAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6);
		}
	}

//	TEST(TestLearnableGlobalPooling, forward_fp32)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor input( { batch_size, height, width, channels });
//		Tensor output( { batch_size, expansion * channels });
//		Tensor weights( { expansion, height * width });
//		testing::initForTest(input, 0.0);
//		testing::initForTest(weights, 1.0);
//
//		Tensor correct_output = zeros_like(output);
//		baseline_learnable_pooling_forward(input, weights, correct_output);
//
//		Tensor tmp_in = input.view( { batch_size, height * width, channels });
//		Tensor tmp_out = output.view( { batch_size, channels, expansion });
//		gemmBatched(context, 't', 't', tmp_out, tmp_in, weights, 1, 0);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			output.moveTo(device);
//			weights.moveTo(device);
//			output.zeroall();
//
//			Tensor tmp_in = input.view( { batch_size, height * width, channels });
//			Tensor tmp_out = output.view( { batch_size, channels, expansion });
//			gemmBatched(context, 't', 't', tmp_out, tmp_in, weights, 1, 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
//		}
//	}
//	TEST(TestLearnableGlobalPooling, backward)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor gradient_prev( { batch_size, height, width, channels });
//		Tensor weights( { expansion, height * width });
//		Tensor gradient_next( { batch_size, expansion * channels });
//		testing::initForTest(weights, 0.0);
//		testing::initForTest(gradient_next, 1.57);
//
//		Tensor correct_gradient_prev = zeros_like(gradient_prev);
//		baseline_learnable_pooling_backward(correct_gradient_prev, weights, gradient_next);
//
//		Tensor tmp_prev = gradient_prev.view( { batch_size, height * width, channels });
//		Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//
//		gemmBatched(context, 't', 't', tmp_prev, weights, tmp_next, 1, 0);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			gradient_prev.moveTo(device);
//			weights.moveTo(device);
//			gradient_next.moveTo(device);
//			gradient_prev.zeroall();
//
//			Tensor tmp_prev = gradient_prev.view( { batch_size, height * width, channels });
//			Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//			gemmBatched(context, 't', 't', tmp_prev, weights, tmp_next, 1, 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//		}
//	}
//	TEST(TestLearnableGlobalPooling, update)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor input( { batch_size, height, width, channels });
//		Tensor gradient_next( { batch_size, expansion * channels });
//		Tensor weights_update( { expansion, height * width });
//		testing::initForTest(input, 0.0);
//		testing::initForTest(gradient_next, 1.57);
//
//		Tensor workspace( { weights_update.volume() * input.firstDim() });
//		Tensor correct_weights_update = zeros_like(weights_update);
//		baseline_learnable_pooling_update(input, correct_weights_update, gradient_next);
//
//		Tensor tmp_in = input.view( { batch_size, height * width, channels });
//		Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//		Tensor tmp_update = workspace.view( { batch_size, expansion, height * width });
//		gemmBatched(context, 't', 't', tmp_update, tmp_next, tmp_in, 1, 0);
////		sumOverFirstDim(context, weights_update, tmp_update.view( { batch_size, expansion * height * width }), 0);
//
//		EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-5f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			gradient_next.moveTo(device);
//			weights_update.moveTo(device);
//			workspace.moveTo(device);
//			weights_update.zeroall();
//
//			Tensor tmp_in = input.view( { batch_size, height * width, channels });
//			Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//			Tensor tmp_update = workspace.view( { batch_size, expansion, height * width });
//			gemmBatched(context, 't', 't', tmp_update, tmp_next, tmp_in, 1, 0);
////			sumOverFirstDim(context, weights_update, tmp_update.view( { batch_size, expansion * height * width }), 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-5f);
//		}
//	}
//	TEST(TestGlobalBroadcasting, forward_fp32)
//	{
//		Context context;
//		Tensor input( { 12, 13, 14, 34 }, "float32", Device::cpu());
//		Tensor bias( { input.firstDim(), input.lastDim() }, "float32", Device::cpu());
//		Tensor output(input.shape(), "float32", Device::cpu());
//		Tensor correct_output(output.shape(), "float32", Device::cpu());
//
//		testing::initForTest(input, 0.0);
//		testing::initForTest(bias, 1.0);
//
//		baseline_broadcasting_forward(input, correct_output, bias, ActivationType::SIGMOID);
//
//		globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
//
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			bias.moveTo(device);
//			output.moveTo(device);
//			output.zeroall();
//			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
//		}
//	}
//	TEST(TestGlobalBroadcasting, backward)
//	{
//		Context context;
//		Tensor output( { 12, 13, 14, 34 }, "float32", Device::cpu());
//		Tensor gradient_next(output.shape(), "float32", Device::cpu());
//		Tensor gradient_prev( { output.firstDim(), output.lastDim() }, "float32", Device::cpu());
//		Tensor correct_gradient_prev(gradient_prev.shape(), "float32", Device::cpu());
//
//		testing::initForTest(gradient_next, 0.0);
//		testing::initForTest(output, 0.0);
//
//		baseline_broadcasting_backward(correct_gradient_prev, gradient_next, output, ActivationType::SIGMOID);
//
//		testing::initForTest(gradient_next, 0.0);
//		globalBroadcastingBackward(context, gradient_prev, gradient_next, output, ActivationType::SIGMOID, 0.0f);
//
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			output.moveTo(device);
//			testing::initForTest(gradient_next, 0.0);
//			gradient_next.moveTo(device);
//			gradient_prev.moveTo(device);
//			gradient_prev.zeroall();
//			globalBroadcastingBackward(context, gradient_prev, gradient_next, output, ActivationType::SIGMOID, 0.0f);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//		}
//	}

	TEST(TestChannelScaling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 36 }, "float32", Device::cpu());
		Tensor scales( { input.firstDim(), input.lastDim() }, "float32", Device::cpu());
		Tensor output(input.shape(), "float32", Device::cpu());
		Tensor correct_output(output.shape(), "float32", Device::cpu());

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);

		baseline_channel_scaling_forward(input, correct_output, scales);

//		channelScalingForward(context, input, output, scales);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			channelScalingForward(context, 1.0f, input, scales, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
		}
	}
	TEST(TestChannelScaling, backward)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 });
		Tensor scales( { input.firstDim(), input.lastDim() });
		Tensor gradient_next = zeros_like(input);

		Tensor gradient_input = zeros_like(input);
		Tensor gradient_scales = zeros_like(scales);

		Tensor correct_gradient_input = zeros_like(gradient_input);
		Tensor correct_gradient_scales = zeros_like(gradient_scales);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);
		testing::initForTest(gradient_next, 2.0);

		baseline_channel_scaling_backward(correct_gradient_input, correct_gradient_scales, gradient_next, input, scales);

//		channelScalingBackward(context, gradient_input, gradient_scales, gradient_next, input, scales);
//		EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-6f);
//		EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			gradient_next.moveTo(device);
			gradient_input.moveTo(device);
			gradient_scales.moveTo(device);
			gradient_input.zeroall();
			gradient_scales.zeroall();
			channelScalingBackward(context, 1.0f, gradient_next, input, scales, 0.0f, gradient_input, 0.0f, gradient_scales);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-5f);
			EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-5f);
		}
	}

//	TEST(TestGlobalPooling, forward_fp16)
//	{
//		Context context;
//		Tensor input( { 12, 13, 14, 34 }, "float16", Device::cpu());
//		Tensor output( { input.firstDim(), 2 * input.lastDim() }, input.dtype(), Device::cpu());
//		Tensor correct_output = zeros_like(output);
//
//		testing::initForTest(input, 0.0);
//
//		baseline_pooling_forward(input, correct_output);
//
//		if (Device::cpu().supportsType(DataType::FLOAT16))
//		{
//			globalAvgAndMaxPoolingForward(context, input, output);
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
//		}
//
//		if (testing::has_device_supporting(DataType::FLOAT16))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			testing::initForTest(output, 1.57f);
//			input.moveTo(device);
//			output.moveTo(device);
//			output.zeroall();
//			globalAvgAndMaxPoolingForward(context, input, output);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
//		}
//	}
//	TEST(TestGlobalBroadcasting, forward_fp16)
//	{
//		Context context;
//		Tensor input( { 12, 13, 14, 34 }, "float16", Device::cpu());
//		Tensor bias( { input.firstDim(), input.lastDim() }, "float16", Device::cpu());
//		Tensor output(input.shape(), "float16", Device::cpu());
//		Tensor correct_output(output.shape(), "float16", Device::cpu());
//
//		testing::initForTest(input, 0.0);
//		testing::initForTest(bias, 1.0);
//
//		baseline_broadcasting_forward(input, correct_output, bias, ActivationType::SIGMOID);
//
//		if (Device::cpu().supportsType(DataType::FLOAT16))
//		{
//			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
//		}
//
//		if (testing::has_device_supporting(DataType::FLOAT16))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			bias.moveTo(device);
//			output.moveTo(device);
//			output.zeroall();
//			globalBroadcastingForward(context, input, output, bias, ActivationType::SIGMOID);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
//		}
//	}

} /* namespace ml */

