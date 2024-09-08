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
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_pooling_backward<float>(gradient_prev[0], gradient_next, input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_pooling_backward<double>(gradient_prev[0], gradient_next, input[0], output);
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
	TEST(TestGlobalPooling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 134 }, "float32", Device::cpu());
		Tensor output( { input.firstDim(), 2 * input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

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
		Tensor gradient_prev = zeros_like(input);
		Tensor output( { input.firstDim(), 2 * input.lastDim() }, "float32", Device::cpu());
		Tensor gradient_next = zeros_like(output);

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

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
		Tensor output( { input.firstDim(), 2 * input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

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

