/*
 * test_depthwise_conv2D.cpp
 *
 *  Created on: Jan 25, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/utils/json.hpp>

#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	void baseline_depthwise_conv2D_forward(const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias, float alpha, float beta)
	{
		assert(input.device().isCPU());
		assert(input.rank() == 4);
		assert(output.rank() == 4);
		assert(weight.rank() == 3);
		assert(output.lastDim() == weight.lastDim()); // output filters

		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters = input.dim(3);

		const int kernel_height = weight.dim(0);
		const int kernel_width = weight.dim(1);

		const int pad_h = -(kernel_height - 1) / 2;
		const int pad_w = -(kernel_width - 1) / 2;

		for (int b = 0; b < batch; b++)
			for (int f = 0; f < filters; f++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
					{
						float tmp = 0.0f;
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 and (pad_h + h + i) < height and (pad_w + w + j) >= 0 and (pad_w + w + j) < width)
									tmp += weight.get( { i, j, f }) * input.get( { b, pad_h + h + i, pad_w + w + j, f });
						tmp *= alpha;
						if (not bias.isEmpty())
							tmp += bias.get( { f });
						if (beta != 0.0f)
							tmp += beta * (float) output.at( { b, h, w, f });
						output.at( { b, h, w, f }) = tmp;
					}
	}
	void baseline_depthwise_conv2D_backward(const Tensor &gradient_next, Tensor &gradient_prev, const Tensor &weight, float alpha, float beta)
	{
		assert(gradient_prev.device().isCPU());
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 4);
		assert(weight.rank() == 3);
		assert(gradient_next.lastDim() == weight.lastDim()); // output filters

		const int batch = gradient_prev.dim(0);
		const int height = gradient_prev.dim(1);
		const int width = gradient_prev.dim(2);
		const int filters = gradient_prev.dim(3);

		const int kernel_height = weight.dim(0);
		const int kernel_width = weight.dim(1);

		const int pad_h = -(kernel_height - 1) / 2;
		const int pad_w = -(kernel_width - 1) / 2;

		for (int b = 0; b < batch; b++)
			for (int f = 0; f < filters; f++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
					{
						float tmp = 0.0f;
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 and (pad_h + h + i) < height and (pad_w + w + j) >= 0 and (pad_w + w + j) < width)
									tmp += weight.get( { kernel_height - 1 - i, kernel_width - 1 - j, f })
											* gradient_next.get( { b, pad_h + h + i, pad_w + w + j, f });
						tmp *= alpha;
						if (beta != 0.0f)
							tmp += beta * (float) gradient_prev.at( { b, h, w, f });
						gradient_prev.at( { b, h, w, f }) = tmp;
					}
	}
	void baseline_depthwise_conv2D_update(const Tensor &input, const Tensor &gradient_next, Tensor &weight_update, Tensor &bias_update, float alpha,
			float beta)
	{
		assert(input.device().isCPU());
		assert(input.rank() == 4);
		assert(gradient_next.rank() == 4);
		assert(weight_update.rank() == 3);
		assert(gradient_next.lastDim() == weight_update.lastDim()); // output filters

		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters = input.dim(3);

		const int kernel_height = weight_update.dim(0);
		const int kernel_width = weight_update.dim(1);

		const int pad_h = -(kernel_height - 1) / 2;
		const int pad_w = -(kernel_width - 1) / 2;

		for (int f = 0; f < filters; f++)
			for (int i = 0; i < kernel_height; i++)
				for (int j = 0; j < kernel_width; j++)
				{
					float tmp = 0.0f;
					for (int b = 0; b < batch; b++)
						for (int h = 0; h < height; h++)
							for (int w = 0; w < width; w++)
								if ((pad_h + h + i) >= 0 and (pad_h + h + i) < height and (pad_w + w + j) >= 0 and (pad_w + w + j) < width)
									tmp += input.get( { b, pad_h + h + i, pad_w + w + j, f }) * gradient_next.get( { b, h, w, f });
					tmp *= alpha;
					if (beta != 0.0f)
						tmp += beta * (float) weight_update.at( { i, j, f });
					weight_update.at( { i, j, f }) = tmp;
				}

		if (not bias_update.isEmpty())
		{
			for (int f = 0; f < filters; f++)
			{
				float tmp = 0.0f;
				for (int b = 0; b < batch; b++)
					for (int h = 0; h < height; h++)
						for (int w = 0; w < width; w++)
							tmp += gradient_next.get( { b, h, w, f });
				tmp *= alpha;
				if (beta != 0.0f)
					tmp += beta * (float) bias_update.at( { f });
				bias_update.at( { f }) = tmp;
			}
		}
	}

	class BaselineDepthwiseConv2D: public Layer
	{
			int m_filters = 0;
			int m_kernel_size = 0;
			bool m_use_bias = true;
			int m_height = 0;
			int m_width = 0;
		public:
			BaselineDepthwiseConv2D(int filters, int kernelSize, bool useBias) :
					Layer("linear")
			{
				m_filters = filters;
				m_kernel_size = kernelSize;
				m_use_bias = useBias;
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
				m_height = shapes[0][1];
				m_width = shapes[0][2];
			}
			Shape getOutputShape() const
			{
				return getInputShape();
			}
			Shape getWeightShape() const
			{
				return Shape( { m_kernel_size, m_kernel_size, m_filters });
			}
			Shape getBiasShape() const
			{
				if (m_use_bias)
					return Shape( { m_filters });
				else
					return Shape();
			}
			std::string name() const
			{
				return "BaselineDepthwiseConv2D";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["filters"] = m_filters;
				result["kernel_size"] = m_kernel_size;
				result["use_bias"] = m_use_bias;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineDepthwiseConv2D> result = std::make_unique<BaselineDepthwiseConv2D>(config["filters"], config["kernel_size"],
						config["use_bias"]);
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 1);
				baseline_depthwise_conv2D_forward(input[0], output, getWeights().getParam(), getBias().getParam(), 1.0f, 0.0f);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 1 && gradient_prev.size() == 1);
				baseline_depthwise_conv2D_backward(gradient_next, gradient_prev[0], getWeights().getParam(), 1.0f, beta[0]);
				baseline_depthwise_conv2D_update(input[0], gradient_next, getWeights().getGradient(), getBias().getGradient(), 1.0f, 0.0f);
			}
	};
}

namespace ml
{
//	TEST(TestDepthwiseConv2D, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineDepthwiseConv2D(32, 7, false) };
//		gradcheck.setInputShape(Shape( { 3, 7, 9, 32 }));
//
//		gradcheck.check(100, 1.0e-2, "all", true);
//
//		exit(0);
//	}

//	TEST(TestDepthwiseConv2D, implementation)
//	{
//		BaselineDepthwiseConv2D baseline(64, 7, false);
//		DepthwiseConv2D actual(64, 7, false);
//
//		std::shared_ptr<Context> context = std::make_shared<Context>(Device::cuda(0));
//		const Shape input_shape( { 32, 15, 15, 64 });
//
//		Tensor input(input_shape, "float32", Device::cpu());
//		Tensor correct_output = zeros_like(input);
//		Tensor output = zeros_like(input);
//		Tensor gradient_next(input_shape, "float32", Device::cpu());
//		Tensor correct_gradient_prev = zeros_like(input);
//		Tensor gradient_prev = zeros_like(input);
//
//		ml::testing::initForTest(gradient_next, 0.0);
//		ml::testing::initForTest(input, 0.0);
//
//		baseline.setInputShape( { input_shape });
//		actual.setInputShape( { input_shape });
//
//		baseline.changeContext(context);
//		actual.changeContext(context);
//
//		baseline.init();
//		actual.getWeights().getParam() = baseline.getWeights().getParam();
//
//		baseline.forward( { input }, correct_output);
//		std::vector<Tensor> asdf = { correct_gradient_prev.view() };
//		baseline.backward( { input }, correct_output, asdf, gradient_next);
//
//		input.moveTo(context->device());
//		output.moveTo(context->device());
//		gradient_next.moveTo(context->device());
//		gradient_prev.moveTo(context->device());
//		actual.forward( { input }, output);
//		context->synchronize();
//		std::vector<Tensor> asdf2 = { gradient_prev.view() };
//		actual.backward( { input }, output, asdf2, gradient_next);
//		context->synchronize();
//
//		std::cout << "forward\n";
//		std::cout << ml::testing::normForTest(correct_output) << " vs " << ml::testing::normForTest(output) << '\n';
//		std::cout << "max diff = " << ml::testing::maxAbsDiff(correct_output, output) << '\n';
//		std::cout << "avg diff = " << ml::testing::diffForTest(correct_output, output) << '\n';
//
//		std::cout << "backward\n";
//		std::cout << ml::testing::normForTest(correct_gradient_prev) << " vs " << ml::testing::normForTest(gradient_prev) << '\n';
//		std::cout << "max diff = " << ml::testing::maxAbsDiff(correct_gradient_prev, gradient_prev) << '\n';
//		std::cout << "avg diff = " << ml::testing::diffForTest(correct_gradient_prev, gradient_prev) << '\n';
//
//		std::cout << "weight update\n";
//		std::cout << ml::testing::normForTest(baseline.getWeights().getParam()) << " vs " << ml::testing::normForTest(actual.getWeights().getParam())
//				<< '\n';
//		std::cout << "max diff = " << ml::testing::maxAbsDiff(baseline.getWeights().getParam(), actual.getWeights().getParam()) << '\n';
//		std::cout << "avg diff = " << ml::testing::diffForTest(baseline.getWeights().getParam(), actual.getWeights().getParam()) << '\n';
//
//		std::cout << "bias update\n";
//		std::cout << ml::testing::normForTest(baseline.getBias().getParam()) << " vs " << ml::testing::normForTest(actual.getBias().getParam())
//				<< '\n';
//		std::cout << "max diff = " << ml::testing::maxAbsDiff(baseline.getBias().getParam(), actual.getBias().getParam()) << '\n';
//		std::cout << "avg diff = " << ml::testing::diffForTest(baseline.getBias().getParam(), actual.getBias().getParam()) << '\n';
//
//		exit(0);
//	}

	TEST(TestDepthwiseConv2D, forward)
	{
		Context context(Device::cpu());

		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filters = 36;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor input( { batch_size, height, width, filters }, "float32", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor weights( { kernel, kernel, filters }, "float32", Device::cpu());
		Tensor bias( { filters }, "float32", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(bias, 0.5f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(output, 1.5f);

		Tensor correct_output = output;
		baseline_depthwise_conv2D_forward(input, correct_output, weights, bias, alpha, beta);

//		depthwiseConvForward(context, input, weights, output, bias);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			ml::testing::initForTest(output, 1.5f);

			depthwiseConvForward(context, alpha, input, weights, beta, output, bias);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestDepthwiseConv2D, backward)
	{
		Context context(Device::cpu());

		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filters = 35;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor gradient_prev( { batch_size, height, width, filters }, "float32", Device::cpu());
		Tensor gradient_next = zeros_like(gradient_prev);
		Tensor weights( { kernel, kernel, filters }, "float32", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(gradient_prev, 1.5f);

		Tensor correct_gradient_prev = gradient_prev;
		baseline_depthwise_conv2D_backward(gradient_next, correct_gradient_prev, weights, alpha, beta);

//		depthwiseConvBackward(context, gradient_next, weights, gradient_prev);
//		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_next.moveTo(device);
			gradient_prev.moveTo(device);
			weights.moveTo(device);
			ml::testing::initForTest(gradient_prev, 1.5f);

			depthwiseConvBackward(context, alpha, gradient_next, weights, beta, gradient_prev);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestDepthwiseConv2D, update)
	{
		Context context(Device::cpu());

		const int batch_size = 32;
		const int height = 13;
		const int width = 17;
		const int filters = 35;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor input( { batch_size, height, width, filters }, "float32", Device::cpu());
		Tensor gradient_next = zeros_like(input);
		ml::testing::initForTest(input, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0);

		Tensor correct_weights_update( { kernel, kernel, filters }, "float32", Device::cpu());
		Tensor weights_update = zeros_like(correct_weights_update);
		Tensor bias_update;
		baseline_depthwise_conv2D_update(input, gradient_next, correct_weights_update, bias_update, alpha, beta);

//		depthwiseConvUpdate(context, input, gradient_next, update);
//		EXPECT_LE(ml::testing::diffForTest(correct_update, update), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weights_update.moveTo(device);

			depthwiseConvUpdate(context, alpha, input, gradient_next, beta, weights_update);
			context.synchronize();

			EXPECT_LE(ml::testing::diffForTest(correct_weights_update, weights_update), 1.0e-2f);
		}
	}

	TEST(TestDepthwiseConv2D, forward_fp16)
	{
		Context context(Device::cpu());

		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filters = 36;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor input( { batch_size, height, width, filters }, "float16", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor weights( { kernel, kernel, filters }, "float16", Device::cpu());
		Tensor bias( { filters }, "float16", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(bias, 0.5f);
		ml::testing::initForTest(input, 1.0f);

		Tensor correct_output = zeros_like(output);
		baseline_depthwise_conv2D_forward(input, correct_output, weights, bias, alpha, beta);

//		depthwiseConvForward(context, input, weights, output, bias);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);

			depthwiseConvForward(context, alpha, input, weights, beta, output, bias);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-2f);
		}
	}
	TEST(TestDepthwiseConv2D, backward_fp16)
	{
		Context context(Device::cpu());

		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filters = 35;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor gradient_prev( { batch_size, height, width, filters }, "float16", Device::cpu());
		Tensor gradient_next = zeros_like(gradient_prev);
		Tensor weights( { kernel, kernel, filters }, "float16", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(gradient_prev, 1.5f);

		Tensor correct_gradient_prev = gradient_prev;
		baseline_depthwise_conv2D_backward(gradient_next, correct_gradient_prev, weights, alpha, beta);

		//		depthwiseConvBackward(context, gradient_next, weights, gradient_prev);
		//		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_next.moveTo(device);
			gradient_prev.moveTo(device);
			weights.moveTo(device);
			ml::testing::initForTest(gradient_prev, 1.5f);

			depthwiseConvBackward(context, alpha, gradient_next, weights, beta, gradient_prev);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-2f);
		}
	}
	TEST(TestDepthwiseConv2D, update_fp16)
	{
		Context context(Device::cpu());

		const int batch_size = 5;
		const int height = 11;
		const int width = 13;
		const int filters = 36;
		const int kernel = 7;

		const float alpha = 1.1f;
		const float beta = 0.1f;

		Tensor input( { batch_size, height, width, filters }, "float16", Device::cpu());
		Tensor gradient_next = zeros_like(input);
		ml::testing::initForTest(input, 0.0f, 0.1f);
		ml::testing::initForTest(gradient_next, 1.0, 0.1f);

		Tensor correct_weights_update( { kernel, kernel, filters }, "float16", Device::cpu());
		Tensor weights_update = zeros_like(correct_weights_update);
		Tensor bias_update;
		baseline_depthwise_conv2D_update(input, gradient_next, correct_weights_update, bias_update, alpha, beta);

		if (ml::testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weights_update.moveTo(device);

			depthwiseConvUpdate(context, alpha, input, gradient_next, beta, weights_update);
			context.synchronize();

			EXPECT_LE(ml::testing::diffForTest(correct_weights_update, weights_update), 1.0e-2f);
		}
	}

} /* namespace ml */

