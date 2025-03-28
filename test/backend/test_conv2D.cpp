/*
 * test_conv2D.cpp
 *
 *  Created on: Jan 31, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/json.hpp>

#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	void baseline_conv2D_forward(const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias, const Tensor &add, ActivationType act)
	{
		assert(input.device().isCPU());
		assert(input.dim(3) == weight.dim(3)); // input filters
		assert(output.dim(3) == weight.dim(0)); // output filters

		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters_in = input.dim(3);
		const int filters_out = output.dim(3);

		const int kernel_height = weight.dim(1);
		const int kernel_width = weight.dim(2);

		const int pad_h = -kernel_height / 2; //TODO handle padding
		const int pad_w = -kernel_width / 2; //TODO handle padding

		output.zeroall();
		for (int b = 0; b < batch; b++)
			for (int out = 0; out < filters_out; out++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
					{
						float tmp = 0.0f;
						if (not bias.isEmpty())
							tmp += bias.get( { out });
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
									for (int in = 0; in < filters_in; in++)
										tmp += weight.get( { out, i, j, in }) * input.get( { b, pad_h + h + i, pad_w + w + j, in });
						if (not add.isEmpty())
							tmp += add.get( { b, h, w, out });
						output.set(tmp, { b, h, w, out });
					}
		activationForward(Context(), 1.0f, output, 0.0f, output, act);
	}
	void baseline_conv2D_backward(const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weight, ActivationType act)
	{
		assert(output.device().isCPU());
		const int batch = output.dim(0);
		const int height = output.dim(1);
		const int width = output.dim(2);
		const int filters_in = gradient_prev.dim(3);
		const int filters_out = gradient_next.dim(3);

		const int kernel_height = weight.dim(1);
		const int kernel_width = weight.dim(2);

		const int pad_h = -kernel_height / 2; //TODO handle padding
		const int pad_w = -kernel_width / 2; //TODO handle padding

		activationBackward(Context(), 1.0f, gradient_next, output, 0.0f, gradient_next, act);
		gradient_prev.zeroall();
		for (int b = 0; b < batch; b++)
			for (int out = 0; out < filters_out; out++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
									for (int in = 0; in < filters_in; in++)
									{
										const float grad = gradient_next.get( { b, h, w, out });
										const float we = weight.get( { out, i, j, in });
										const float pr = gradient_prev.get( { b, pad_h + h + i, pad_w + w + j, in });
										gradient_prev.set(pr + grad * we, { b, pad_h + h + i, pad_w + w + j, in });
									}
	}
	void baseline_conv2D_update(const Tensor &input, const Tensor &gradient_next, Tensor &weight_update)
	{
		assert(input.device().isCPU());
		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters_in = input.dim(3);
		const int filters_out = gradient_next.dim(3);

		const int kernel_height = weight_update.dim(1);
		const int kernel_width = weight_update.dim(2);

		const int pad_h = -kernel_height / 2; //TODO handle padding
		const int pad_w = -kernel_width / 2; //TODO handle padding
		weight_update.zeroall();

		for (int b = 0; b < batch; b++)
			for (int out = 0; out < filters_out; out++)
			{
				for (int in = 0; in < filters_in; in++)
					for (int i = 0; i < kernel_height; i++)
						for (int j = 0; j < kernel_width; j++)
						{
							float tmp = weight_update.get( { out, i, j, in });
							for (int h = 0; h < height; h++)
								for (int w = 0; w < width; w++)
									if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
										tmp += gradient_next.get( { b, h, w, out }) * input.get( { b, pad_h + h + i, pad_w + w + j, in });
							weight_update.set(tmp, { out, i, j, in });
						}
			}
	}

//	class BaselineConv2D: public Layer
//	{
//			int m_number_of_heads = 0;
//			int m_positional_encoding_range = 0;
//			bool m_symmetric = false;
//		public:
//			BaselineConv2D(int numberOfHeads, int positional_encoding_range, bool symmetric) :
//					Layer(),
//					m_number_of_heads(numberOfHeads),
//					m_positional_encoding_range(positional_encoding_range),
//					m_symmetric(symmetric)
//			{
//			}
//			Shape getWeightShape() const
//			{
//				if (m_positional_encoding_range > 0)
//				{
//					const int tmp = 2 * m_positional_encoding_range - 1;
//					return Shape( { m_number_of_heads, tmp, round_up(tmp, 4) });
//				}
//				else
//					return Shape();
//			}
//			void setInputShape(const std::vector<Shape> &shapes)
//			{
//				m_input_shapes = shapes;
//			}
//			Shape getOutputShape() const
//			{
//				const int batch_size = getInputShape().dim(0);
//				const int height = getInputShape().dim(1);
//				const int width = getInputShape().dim(2);
//				const int embedding = getInputShape().dim(3) / (3 - m_symmetric);
//				return Shape( { batch_size, height, width, embedding });
//			}
//			std::string name() const
//			{
//				return "BaselineMHA";
//			}
//			Json getConfig() const
//			{
//				Json result = Layer::getConfig();
//				result["number_of_heads"] = m_number_of_heads;
//				result["positional_encoding_range"] = m_positional_encoding_range;
//				result["symmetric"] = m_symmetric;
//				return result;
//			}
//			std::unique_ptr<Layer> clone(const Json &config) const
//			{
//				std::unique_ptr<BaselineMHA> result = std::make_unique<BaselineMHA>(config["number_of_heads"].getInt(),
//						config["positional_encoding_range"].getInt(), config["symmetric"].getBool());
//				result->m_dtype = typeFromString(config["dtype"].getString());
//				return result;
//			}
//
//			void forward(const std::vector<Tensor> &input, Tensor &output)
//			{
//				output = baseline_mha_forward(input[0], getWeights().getParam(), m_number_of_heads, m_symmetric);
//			}
//			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
//			{
//				baseline_mha_backward(input[0], getWeights().getParam(), gradient_prev[0], gradient_next, getWeights().getGradient(),
//						m_number_of_heads, m_symmetric);
//			}
//	};
}

namespace ml
{
//	TEST(TestConv2D, baseline)
//	{
//		ml::testing::GradientCheck gradcheck { BaselineConv2D(2, 5, false) };
//		gradcheck.setInputShape(Shape( { 3, 8, 8, 3 * 32 }));
//
//		gradcheck.check(100, 1.0e-4, "input");
//
//		exit(0);
//	}

	TEST(TestConv2D, explicit_gemm_conv2D_1x1_forward_fp32)
	{
		Context context(Device::cpu());
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int channels_in = 35;
		const int channels_out = 21;
		Tensor input( { batch_size, height, width, channels_in });
		Tensor output( { batch_size, height, width, channels_out });
		Tensor weights( { channels_out, 1, 1, channels_in }, "float32", Device::cpu());
		Tensor bias( { channels_out });
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, Tensor(), ActivationType::SIGMOID);

		Tensor weight_matrices = weights.view(Shape( { channels_out, 1 * 1 * channels_in }));
		Tensor input_matrices = input.view(Shape( { batch_size * height * width, channels_in }));
		Tensor output_matrices = output.view(Shape( { batch_size * height * width, channels_out }));

		gemm_ex(context, output_matrices, 1.0f, 'n', input_matrices, 't', weight_matrices, 0.0f, output_matrices, bias, ActivationType::SIGMOID);
		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			Tensor weight_matrices = weights.view(Shape( { channels_out, 1 * 1 * channels_in }));
			Tensor input_matrices = input.view(Shape( { batch_size * height * width, channels_in }));
			Tensor output_matrices = output.view(Shape( { batch_size * height * width, channels_out }));

			output_matrices.zeroall();

			gemm_ex(context, output_matrices, 1.0f, 'n', input_matrices, 't', weight_matrices, 0.0f, output_matrices, bias, ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_1x1_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 3, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 3, 13, 17, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 1, 1, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		Tensor weight_matrices = weights.view( { 21, 1 * 1 * 35 });
		Tensor gradient_prev_matrices = gradient_prev.view( { 3 * 13 * 17, 35 });
		Tensor gradient_next_matrices = gradient_next.view( { 3 * 13 * 17, 21 });

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);
		gemm(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);

			gradient_prev.zeroall();

			Tensor weight_matrices = weights.view( { 21, 1 * 1 * 35 });
			Tensor gradient_prev_matrices = gradient_prev.view( { 3 * 13 * 17, 35 });
			Tensor gradient_next_matrices = gradient_next.view( { 3 * 13 * 17, 21 });

			ml::testing::initForTest(gradient_next, 1.0f);
			activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);
			gemm(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_1x1_update)
	{
		Context context(Device::cpu());
		Tensor input( { 3, 13, 17, 35 }, "float32", Device::cpu());
		Tensor gradient_next( { 3, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weight_update( { 21, 1, 1, 35 }, "float32", Device::cpu());
		ml::testing::initForTest(input, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrix = weight_update.view( { 21, 35 });
		Tensor input_matrix = input.view( { 3 * 13 * 17, 35 });
		Tensor gradient_next_matrix = gradient_next.view( { 3 * 13 * 17, 21 });

		gemm(context, 't', 'n', weight_update_matrix, gradient_next_matrix, input_matrix, 1, 0);
		EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			ml::testing::initForTest(weight_update, 1.57f);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weight_update.moveTo(device);

			Tensor weight_update_matrix = weight_update.view( { 21, 35 });
			Tensor input_matrix = input.view( { 3 * 13 * 17, 35 });
			Tensor gradient_next_matrix = gradient_next.view( { 3 * 13 * 17, 21 });

			gemm(context, 't', 'n', weight_update_matrix, gradient_next_matrix, input_matrix, 1, 0);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_5x5_forward_fp32)
	{
		Context context(Device::cpu());
		Tensor input( { 2, 5, 5, 35 });
		Tensor output( { 2, 5, 5, 21 });
		Tensor weights( { output.lastDim(), 5, 5, input.lastDim() });
		Tensor bias( { 21 });
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);

		Tensor correct_output = zeros_like(output);
		baseline_conv2D_forward(input, correct_output, weights, bias, Tensor(), ActivationType::RELU);

		std::array<int, 3> workspace_size = explicit_gemm_workspace(input.shape(), output.shape(), weights.shape());
		Tensor workspace( { workspace_size[0] });

		explicit_gemm_forward(context, input, output, weights, bias, workspace, ActivationType::RELU, Tensor());
		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			workspace.moveTo(device);

			output.zeroall();

			explicit_gemm_forward(context, input, output, weights, bias, workspace, ActivationType::RELU, Tensor());
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_5x5_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 3, 11, 12, 35 });
		Tensor output( { 3, 11, 12, 21 });
		Tensor gradient_next = zeros_like(output);
		Tensor weights( { output.lastDim(), 5, 5, gradient_prev.lastDim() });
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev = zeros_like(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::LINEAR);

		std::array<int, 3> workspace_size = explicit_gemm_workspace(gradient_prev.shape(), output.shape(), weights.shape());
		Tensor workspace( { workspace_size[1] });

//		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);
			workspace.moveTo(device);

			gradient_prev.zeroall();

			explicit_gemm_backward(context, gradient_prev, gradient_next, output, weights, workspace, 0.0f);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_5x5_update)
	{
		Context context(Device::cpu());
		Tensor input( { 3, 11, 12, 35 });
		Tensor gradient_next( { 3, 11, 12, 21 });
		Tensor weight_update( { 21, 5, 5, 35 });
		ml::testing::initForTest(input, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);

		Tensor correct_weight_update = zeros_like(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		std::array<int, 3> workspace_size = explicit_gemm_workspace(input.shape(), gradient_next.shape(), weight_update.shape());
		Tensor workspace( { workspace_size[2] });

//		EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weight_update.moveTo(device);
			workspace.moveTo(device);

			explicit_gemm_update(context, input, gradient_next, weight_update, workspace);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

#ifdef USE_CUDNN
	TEST(TestConv2D, implicit_gemm_conv2D_1x1_forward_fp16)
	{
		Context context(Device::cpu());
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 1, 1, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);
		ml::testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (Device::numberOfCudaDevices() > 0 and Device::cuda(0).supportsType(DataType::FLOAT16))
		{
			Context context(Device::cuda(0));
			input.moveTo(context.device());
			output.moveTo(context.device());
			add.moveTo(context.device());
			weights.moveTo(context.device());
			bias.moveTo(context.device());

			output.zeroall();

			convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
			output.convertTo(context, DataType::FLOAT32);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-1f);

//			for (int i = 0; i < height; i++)
//			{
//				for (int j = 0; j < width; j++)
//					std::cout << correct_output.get( { 0, i, j, 0 }) << ' ';
//				std::cout << '\n';
//			}
//			std::cout << "-----------------------------------------------------------\n";
//			for (int i = 0; i < height; i++)
//			{
//				for (int j = 0; j < width; j++)
//					std::cout << output.get( { 0, i, j, 0 }) << ' ';
//				std::cout << '\n';
//			}
		}
	}
	TEST(TestConv2D, implicit_gemm_conv2D_3x3_forward_fp16)
	{
		Context context(Device::cpu());
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 3, 3, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);
		ml::testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (Device::numberOfCudaDevices() > 0 and Device::cuda(0).supportsType(DataType::FLOAT16))
		{
			Context context(Device::cuda(0));
			input.moveTo(context.device());
			output.moveTo(context.device());
			add.moveTo(context.device());
			weights.moveTo(context.device());
			bias.moveTo(context.device());

			output.zeroall();

			convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
			output.convertTo(context, DataType::FLOAT32);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-1f);
		}
	}
	TEST(TestConv2D, implicit_gemm_conv2D_5x5_forward_fp16)
	{
		Context context(Device::cpu());
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 5, 5, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);
		ml::testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (Device::numberOfCudaDevices() > 0 and Device::cuda(0).supportsType(DataType::FLOAT16))
		{
			Context context(Device::cuda(0));
			input.moveTo(context.device());
			output.moveTo(context.device());
			add.moveTo(context.device());
			weights.moveTo(context.device());
			bias.moveTo(context.device());

			output.zeroall();

			convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
			output.convertTo(context, DataType::FLOAT32);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-1f);
		}
	}
#endif

	TEST(TestConv2D, winograd_conv2D_3x3_forward)
	{
		Context context(Device::cpu());
		Tensor input( { 3, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 3, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weights( { 21, 3, 3, 35 }, "float32", Device::cpu());
		Tensor bias( { 21 }, "float32", Device::cpu());
		Tensor add( { 3, 13, 17, 21 }, "float32", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);
		ml::testing::initForTest(input, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		winogradWeightTransform(context, weights, weight_matrices, false);

		Tensor input_matrices( { 36, 3 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor output_matrices( { 36, 3 * 4 * 5, 21 }, "float32", Device::cpu());
		winogradInputTransform(context, weights.shape(), input, input_matrices);
		gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1, 0);
		winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID, 0.0f);
		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			weight_matrices.moveTo(device);
			input_matrices.moveTo(device);
			output_matrices.moveTo(device);

			weight_matrices.zeroall();
			input_matrices.zeroall();
			output_matrices.zeroall();

			winogradWeightTransform(context, weights, weight_matrices, false);
			winogradInputTransform(context, weights.shape(), input, input_matrices);
			gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1, 0);
			winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID, 0.0f);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_3x3_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 3, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 3, 13, 17, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 3, 3, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 3 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 3 * 4 * 5, 21 }, "float32", Device::cpu());

		winogradWeightTransform(context, weights, weight_matrices, true);
		winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
		gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
		winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR, 0.0f);
		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);
			weight_matrices.moveTo(device);
			gradient_prev_matrices.moveTo(device);
			gradient_next_matrices.moveTo(device);

			weight_matrices.zeroall();
			gradient_prev_matrices.zeroall();
			gradient_next_matrices.zeroall();

			ml::testing::initForTest(gradient_next, 1.0f);
			activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);
			winogradWeightTransform(context, weights, weight_matrices, true);
			winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
			gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
			winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR,
					0.0f);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_3x3_update)
	{
		Context context(Device::cpu());
		Tensor input( { 3, 13, 17, 35 }, "float32", Device::cpu());
		Tensor gradient_next( { 3, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weight_update( { 21, 3, 3, 35 }, "float32", Device::cpu());
		Tensor storage( { 8, 21 }, "float32", Device::cpu());
		ml::testing::initForTest(input, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 3 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 3 * 4 * 5, 21 }, "float32", Device::cpu());

		winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
		winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
		gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
		winogradUpdateTransform(context, weight_update_matrices, weight_update);

		EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			ml::testing::initForTest(weight_update, 1.57f);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weight_update.moveTo(device);
			storage.moveTo(device);
			weight_update_matrices.moveTo(device);
			gradient_prev_matrices.moveTo(device);
			gradient_next_matrices.moveTo(device);

			weight_update_matrices.zeroall();
			gradient_prev_matrices.zeroall();
			gradient_next_matrices.zeroall();

			winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
			winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
			gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
			winogradUpdateTransform(context, weight_update_matrices, weight_update);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

	TEST(TestConv2D, winograd_conv2D_5x5_forward)
	{
		Context context(Device::cpu());
		Tensor input( { 3, 11, 13, 35 }, "float32", Device::cpu());
		Tensor output( { 3, 11, 13, 21 }, "float32", Device::cpu());
		Tensor weights( { 21, 5, 5, 35 }, "float32", Device::cpu());
		Tensor bias( { 21 }, "float32", Device::cpu());
		ml::testing::initForTest(weights, 0.0f);
		ml::testing::initForTest(input, 1.0f);
		ml::testing::initForTest(bias, 1.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, Tensor(), ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		winogradWeightTransform(context, weights, weight_matrices, false);

		Tensor input_matrices( { 36, 3 * 6 * 7, 35 }, "float32", Device::cpu());
		Tensor output_matrices( { 36, 3 * 6 * 7, 21 }, "float32", Device::cpu());
		winogradInputTransform(context, weights.shape(), input, input_matrices);
		gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1.0f, 0.0f);
		winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID, 0.0f);
		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			weight_matrices.moveTo(device);
			input_matrices.moveTo(device);
			output_matrices.moveTo(device);

			weight_matrices.zeroall();
			input_matrices.zeroall();
			output_matrices.zeroall();

			winogradWeightTransform(context, weights, weight_matrices, false);
			winogradInputTransform(context, weights.shape(), input, input_matrices);
			gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1.0f, 0.0f);
			winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID, 0.0f);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_5x5_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 3, 11, 13, 35 }, "float32", Device::cpu());
		Tensor output( { 3, 11, 13, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 5, 5, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 3 * 6 * 7, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 3 * 6 * 7, 21 }, "float32", Device::cpu());

		winogradWeightTransform(context, weights, weight_matrices, true);
		winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
		gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1.0f, 0.0f);
		winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR, 0.0f);
		EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);
			weight_matrices.moveTo(device);
			gradient_prev_matrices.moveTo(device);
			gradient_next_matrices.moveTo(device);

			weight_matrices.zeroall();
			gradient_prev_matrices.zeroall();
			gradient_next_matrices.zeroall();

			ml::testing::initForTest(gradient_next, 1.0f);
			activationBackward(context, 1.0f, gradient_next, output, 0.0f, gradient_next, ActivationType::SIGMOID);
			winogradWeightTransform(context, weights, weight_matrices, true);
			winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
			gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1.0f, 0.0f);
			winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR,
					0.0f);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_5x5_update)
	{
		const int batch_size = 3;
		const int height = 11;
		const int width = 13;
		const int filters_in = 21;
		const int filters_out = 34;

		Context context(Device::cpu());
		Tensor input( { batch_size, height, width, filters_in }, "float32", Device::cpu());
		Tensor gradient_next( { batch_size, height, width, filters_out }, "float32", Device::cpu());
		Tensor weight_update( { filters_out, 5, 5, filters_in }, "float32", Device::cpu());
		ml::testing::initForTest(input, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrices( { 36, filters_out, filters_in }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, batch_size * ((height + 1) / 2) * ((width + 1) / 2), filters_in }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, batch_size * ((height + 1) / 2) * ((width + 1) / 2), filters_out }, "float32", Device::cpu());

		winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
		winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
		gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
		winogradUpdateTransform(context, weight_update_matrices, weight_update);

		EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			ml::testing::initForTest(weight_update, 1.57f);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weight_update.moveTo(device);
			weight_update_matrices.moveTo(device);
			gradient_prev_matrices.moveTo(device);
			gradient_next_matrices.moveTo(device);

			weight_update_matrices.zeroall();
			gradient_prev_matrices.zeroall();
			gradient_next_matrices.zeroall();

			winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
			winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
			gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
			winogradUpdateTransform(context, weight_update_matrices, weight_update);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

} /* namespace ml */
