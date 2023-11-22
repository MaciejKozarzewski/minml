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

#include <gtest/gtest.h>

namespace
{
	void baseline_conv2D_forward(const ml::Tensor &input, ml::Tensor &output, const ml::Tensor &weight, const ml::Tensor &bias, const ml::Tensor &add,
			ml::ActivationType act)
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
		ml::activationForward(ml::Context(), output, output, act);
	}
	void baseline_conv2D_backward(const ml::Tensor &output, ml::Tensor &gradient_prev, ml::Tensor &gradient_next, const ml::Tensor &weight,
			ml::ActivationType act)
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

		ml::activationBackward(ml::Context(), gradient_next, gradient_next, output, act);
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
	void baseline_conv2D_update(const ml::Tensor &input, const ml::Tensor &gradient_next, ml::Tensor &weight_update)
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
}

namespace ml
{
	TEST(TestConv2D, explicit_gemm_conv2D_1x1_forward_fp32)
	{
		Context context(Device::cpu());
		Tensor input( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weights( { 21, 1, 1, 35 }, "float32", Device::cpu());
		Tensor bias( { 21 }, "float32", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, Tensor(), ActivationType::SIGMOID);

		Tensor weight_matrices = weights.view(Shape( { 21, 1 * 1 * 35 }));
		Tensor input_matrices = input.view(Shape( { 12 * 13 * 17, 35 }));
		Tensor output_matrices = output.view(Shape( { 12 * 13 * 17, 21 }));

		gemm(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1.0f, 0.0f);
		addBiasAct(context, output, output, bias, ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			Tensor weight_matrices = weights.view(Shape( { 21, 1 * 1 * 35 }));
			Tensor input_matrices = input.view(Shape( { 12 * 13 * 17, 35 }));
			Tensor output_matrices = output.view(Shape( { 12 * 13 * 17, 21 }));

			output_matrices.zeroall();

			gemm(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1.0f, 0.0f);
			addBiasAct(context, output, output, bias, ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_1x1_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 1, 1, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		Tensor weight_matrices = weights.view( { 21, 1 * 1 * 35 });
		Tensor gradient_prev_matrices = gradient_prev.view( { 12 * 13 * 17, 35 });
		Tensor gradient_next_matrices = gradient_next.view( { 12 * 13 * 17, 21 });

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);
		gemm(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			output.moveTo(device);
			gradient_next.moveTo(device);
			weights.moveTo(device);

			gradient_prev.zeroall();

			Tensor weight_matrices = weights.view( { 21, 1 * 1 * 35 });
			Tensor gradient_prev_matrices = gradient_prev.view( { 12 * 13 * 17, 35 });
			Tensor gradient_next_matrices = gradient_next.view( { 12 * 13 * 17, 21 });

			ml::testing::initForTest(gradient_next, 1.0f);
			activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);
			gemm(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, explicit_gemm_conv2D_1x1_update)
	{
		Context context(Device::cpu());
		Tensor input( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor gradient_next( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weight_update( { 21, 1, 1, 35 }, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.0f);
		testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrix = weight_update.view( { 21, 35 });
		Tensor input_matrix = input.view( { 12 * 13 * 17, 35 });
		Tensor gradient_next_matrix = gradient_next.view( { 12 * 13 * 17, 21 });

		gemm(context, 't', 'n', weight_update_matrix, gradient_next_matrix, input_matrix, 1, 0);
		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 1.57f);
			input.moveTo(device);
			gradient_next.moveTo(device);
			weight_update.moveTo(device);

			Tensor weight_update_matrix = weight_update.view( { 21, 35 });
			Tensor input_matrix = input.view( { 12 * 13 * 17, 35 });
			Tensor gradient_next_matrix = gradient_next.view( { 12 * 13 * 17, 21 });

			gemm(context, 't', 'n', weight_update_matrix, gradient_next_matrix, input_matrix, 1, 0);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

#ifdef USE_CUDNN
	TEST(TestConv2D, implicit_gemm_conv2D_1x1_forward_fp16)
	{
		Context context(Device::cpu());
		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 1, 1, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);
		testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

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
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-1f);

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
		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 3, 3, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);
		testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

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
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-1f);
		}
	}
	TEST(TestConv2D, implicit_gemm_conv2D_5x5_forward_fp16)
	{
		Context context(Device::cpu());
		const int batch_size = 12;
		const int height = 13;
		const int width = 17;
		const int filter_in = 35;
		const int filter_out = 21;

		Tensor input( { batch_size, height, width, filter_in }, "float16", Device::cpu());
		Tensor output( { batch_size, height, width, filter_out }, "float16", Device::cpu());
		Tensor add(output.shape(), "float16", Device::cpu());
		Tensor weights( { filter_out, 5, 5, filter_in }, "float16", Device::cpu());
		Tensor bias( { filter_out }, "float16", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);
		testing::initForTest(add, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::RELU);

//		convolutionImplicitGemmForward(context, input, weights, output, bias, add, ActivationType::RELU);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

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
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-1f);
		}
	}
#endif

	TEST(TestConv2D, winograd_conv2D_3x3_forward)
	{
		Context context(Device::cpu());
		Tensor input( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weights( { 21, 3, 3, 35 }, "float32", Device::cpu());
		Tensor bias( { 21 }, "float32", Device::cpu());
		Tensor add( { 12, 13, 17, 21 }, "float32", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);
		testing::initForTest(input, 2.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, add, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		winogradWeightTransform(context, weights, weight_matrices, false);

		Tensor input_matrices( { 36, 12 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor output_matrices( { 36, 12 * 4 * 5, 21 }, "float32", Device::cpu());
		winogradInputTransform(context, weights.shape(), input, input_matrices);
		gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1, 0);
		winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
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
			winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_3x3_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 3, 3, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 12 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 12 * 4 * 5, 21 }, "float32", Device::cpu());

		winogradWeightTransform(context, weights, weight_matrices, true);
		winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
		gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
		winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR);
		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
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
			activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);
			winogradWeightTransform(context, weights, weight_matrices, true);
			winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
			gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1, 0);
			winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_3x3_update)
	{
		Context context(Device::cpu());
		Tensor input( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor gradient_next( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weight_update( { 21, 3, 3, 35 }, "float32", Device::cpu());
		Tensor storage( { 8, 21 }, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.0f);
		testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 12 * 4 * 5, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 12 * 4 * 5, 21 }, "float32", Device::cpu());

		winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
		winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
		gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
		winogradUpdateTransform(context, weight_update_matrices, weight_update);

		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 1.57f);
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
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

	TEST(TestConv2D, winograd_conv2D_5x5_forward)
	{
		Context context(Device::cpu());
		Tensor input( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor weights( { 21, 5, 5, 35 }, "float32", Device::cpu());
		Tensor bias( { 21 }, "float32", Device::cpu());
		testing::initForTest(weights, 0.0f);
		testing::initForTest(input, 1.0f);
		testing::initForTest(bias, 1.0f);

		Tensor correct_output(output.shape(), "float32", Device::cpu());
		baseline_conv2D_forward(input, correct_output, weights, bias, Tensor(), ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		winogradWeightTransform(context, weights, weight_matrices, false);

		Tensor input_matrices( { 36, 12 * 7 * 9, 35 }, "float32", Device::cpu());
		Tensor output_matrices( { 36, 12 * 7 * 9, 21 }, "float32", Device::cpu());
		winogradInputTransform(context, weights.shape(), input, input_matrices);
		gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1.0f, 0.0f);
		winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
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
			winogradOutputTransform(context, weights.shape(), output_matrices, output, bias, Tensor(), ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_5x5_backward)
	{
		Context context(Device::cpu());
		Tensor gradient_prev( { 12, 13, 17, 35 }, "float32", Device::cpu());
		Tensor output( { 12, 13, 17, 21 }, "float32", Device::cpu());
		Tensor gradient_next(output.shape(), "float32", Device::cpu());
		Tensor weights( { output.lastDim(), 5, 5, gradient_prev.lastDim() }, "float32", Device::cpu());
		ml::testing::initForTest(output, 0.0f);
		ml::testing::initForTest(gradient_next, 1.0f);
		ml::testing::initForTest(weights, 1.57f);

		Tensor correct_gradient_prev(gradient_prev);
		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, ActivationType::SIGMOID);

		ml::testing::initForTest(gradient_next, 1.0f);
		activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);

		Tensor weight_matrices( { 36, 21, 35 }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, 12 * 7 * 9, 35 }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, 12 * 7 * 9, 21 }, "float32", Device::cpu());

		winogradWeightTransform(context, weights, weight_matrices, true);
		winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
		gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1.0f, 0.0f);
		winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR);
		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
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
			activationBackward(context, gradient_next, gradient_next, output, ActivationType::SIGMOID);
			winogradWeightTransform(context, weights, weight_matrices, true);
			winogradInputTransform(context, weights.shape(), gradient_next, gradient_next_matrices);
			gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices, 1.0f, 0.0f);
			winogradOutputTransform(context, weights.shape(), gradient_prev_matrices, gradient_prev, Tensor(), Tensor(), ActivationType::LINEAR);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
	TEST(TestConv2D, winograd_conv2D_5x5_update)
	{
		const int batch_size = 12;
		const int height = 15;
		const int width = 15;
		const int filters_in = 21;
		const int filters_out = 34;

		Context context(Device::cpu());
		Tensor input( { batch_size, height, width, filters_in }, "float32", Device::cpu());
		Tensor gradient_next( { batch_size, height, width, filters_out }, "float32", Device::cpu());
		Tensor weight_update( { filters_out, 5, 5, filters_in }, "float32", Device::cpu());
		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.0f);
		testing::initForTest(weight_update, 1.57f);

		Tensor correct_weight_update(weight_update);
		baseline_conv2D_update(input, gradient_next, correct_weight_update);

		Tensor weight_update_matrices( { 36, filters_out, filters_in }, "float32", Device::cpu());
		Tensor gradient_prev_matrices( { 36, batch_size * ((height + 1) / 2) * ((width + 1) / 2), filters_in }, "float32", Device::cpu());
		Tensor gradient_next_matrices( { 36, batch_size * ((height + 1) / 2) * ((width + 1) / 2), filters_out }, "float32", Device::cpu());

		winogradGradientTransform(context, weight_update.shape(), gradient_next, gradient_next_matrices);
		winogradInputTransform(context, weight_update.shape(), input, gradient_prev_matrices);
		gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1.0f, 0.0f);
		winogradUpdateTransform(context, weight_update_matrices, weight_update);

		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(weight_update, 1.57f);
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
			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
		}
	}

} /* namespace ml */
