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
	void baseline_conv2D_forward(const ml::Tensor &input, ml::Tensor &output, const ml::Tensor &weight, const ml::Tensor &bias,
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

		output.zeroall(ml::Context());
		for (int b = 0; b < batch; b++)
			for (int out = 0; out < filters_out; out++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
					{
						float tmp = 0.0f;
						if (!bias.isEmpty())
							tmp += bias.get( { out });
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
									for (int in = 0; in < filters_in; in++)
										tmp += weight.get( { out, i, j, in }) * input.get( { b, pad_h + h + i, pad_w + w + j, in });
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
		gradient_prev.zeroall(ml::Context());
		for (int b = 0; b < batch; b++)
			for (int out = 0; out < filters_out; out++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
								if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
									for (int in = 0; in < filters_in; in++)
									{
										float grad = gradient_next.get( { b, h, w, out });
										float we = weight.get( { out, i, j, in });
										float pr = gradient_prev.get( { b, pad_h + h + i, pad_w + w + j, in });
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

	int max(std::tuple<int, int, int> t)
	{
		return std::max(std::get<0>(t), std::max(std::get<1>(t), std::get<2>(t)));
	}
}

namespace ml
{
//	TEST(TestConv2D, explicit_gemm_conv2D_3x3_forward)
//	{
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor output( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weights( { 21, 3, 3, 35 }, "float32", Device::cpu());
//		Tensor bias( { 21 }, "float32", Device::cpu());
//		testing::initForTest(weights, 0.0f);
//		testing::initForTest(input, 1.0f);
//		testing::initForTest(bias, 1.0f);
//
//		Tensor correct_output(output.shape(), "float32", Device::cpu());
//		baseline_conv2D_forward(input, correct_output, weights, bias, ActivationType::RELU);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(input.shape(), output.shape(), weights.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmForward(DeviceContext(), input, output, weights, bias, workspace, NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//			weights.moveTo(Device::cuda(0));
//			bias.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//			output.zeroall(context);
//
//			math::convolution2D::explicitGemmForward(context, input, output, weights, bias, workspace, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//		}
//	}
//	TEST(TestConv2D, explicit_gemm_conv2D_3x3_backward)
//	{
//		Tensor gradient_prev( { 2, 16, 4, 5 }, "float32", Device::cpu());
//		Tensor output( { 2, 16, 4, 15 }, "float32", Device::cpu());
//		Tensor gradient_next(output.shape(), "float32", Device::cpu());
//		Tensor weights( { output.lastDim(), 3, 3, gradient_prev.lastDim() }, "float32", Device::cpu());
//		testing::initForTest(output, 0.0f);
//		testing::initForTest(gradient_next, 1.0f);
//		testing::initForTest(weights, 1.57f);
//
//		Tensor correct_gradient_prev(gradient_prev);
//		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, NonlinearityType::SIGMOID);
//		testing::initForTest(gradient_next, 1.0f);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(gradient_prev.shape(), output.shape(), weights.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmBackward(DeviceContext(), gradient_prev, gradient_next, output, weights, workspace,
//				NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			gradient_prev.moveTo(Device::cuda(0));
//			gradient_prev.zeroall(context);
//			gradient_next.moveTo(Device::cuda(0));
//			testing::initForTest(gradient_next, 1.0f);
//			weights.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//
//			math::convolution2D::explicitGemmBackward(context, gradient_prev, gradient_next, output, weights, workspace, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//		}
//	}
//	TEST(TestConv2D, explicit_gemm_conv2D_3x3_update)
//	{
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor gradient_next( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weight_update( { 21, 3, 3, 35 }, "float32", Device::cpu());
//		Tensor storage( { 8, 21 }, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(gradient_next, 1.0f);
//		testing::initForTest(weight_update, 1.57f);
//
//		Tensor correct_weight_update(weight_update);
//		baseline_conv2D_update(input, gradient_next, correct_weight_update);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(input.shape(), gradient_next.shape(), weight_update.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmUpdate(DeviceContext(), input, gradient_next, weight_update, workspace);
//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			storage.moveTo(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			gradient_next.moveTo(Device::cuda(0));
//			testing::initForTest(weight_update, 1.57f);
//			weight_update.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//
//			math::convolution2D::explicitGemmUpdate(context, input, gradient_next, weight_update, workspace);
//			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//		}
//	}
//
//	TEST(TestConv2D, explicit_gemm_conv2D_5x5_forward)
//	{
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor output( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weights( { 21, 5, 5, 35 }, "float32", Device::cpu());
//		Tensor bias( { 21 }, "float32", Device::cpu());
//		testing::initForTest(weights, 0.0f);
//		testing::initForTest(input, 1.0f);
//		testing::initForTest(bias, 1.0f);
//
//		Tensor correct_output(output.shape(), "float32", Device::cpu());
//		baseline_conv2D_forward(input, correct_output, weights, bias, NonlinearityType::SIGMOID);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(input.shape(), output.shape(), weights.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmForward(DeviceContext(), input, output, weights, bias, workspace, NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//			weights.moveTo(Device::cuda(0));
//			bias.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//			output.zeroall(context);
//
//			math::convolution2D::explicitGemmForward(context, input, output, weights, bias, workspace, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//		}
//	}
//	TEST(TestConv2D, explicit_gemm_conv2D_5x5_backward)
//	{
//		Tensor gradient_prev( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor output( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor gradient_next(output.shape(), "float32", Device::cpu());
//		Tensor weights( { output.lastDim(), 5, 5, gradient_prev.lastDim() }, "float32", Device::cpu());
//		testing::initForTest(output, 0.0f);
//		testing::initForTest(gradient_next, 1.0f);
//		testing::initForTest(weights, 1.57f);
//
//		Tensor correct_gradient_prev(gradient_prev);
//		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, NonlinearityType::SIGMOID);
//		testing::initForTest(gradient_next, 1.0f);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(gradient_prev.shape(), output.shape(), weights.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmBackward(DeviceContext(), gradient_prev, gradient_next, output, weights, workspace,
//				NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			gradient_prev.moveTo(Device::cuda(0));
//			gradient_prev.zeroall(context);
//			gradient_next.moveTo(Device::cuda(0));
//			testing::initForTest(gradient_next, 1.0f);
//			weights.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//			output.moveTo(Device::cuda(0));
//
//			math::convolution2D::explicitGemmBackward(context, gradient_prev, gradient_next, output, weights, workspace, NonlinearityType::SIGMOID);
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//		}
//	}
//	TEST(TestConv2D, explicit_gemm_conv2D_5x5_update)
//	{
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor gradient_next( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weight_update( { 21, 5, 5, 35 }, "float32", Device::cpu());
//		Tensor storage( { 8, 21 }, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(gradient_next, 1.0f);
//		testing::initForTest(weight_update, 1.57f);
//
//		Tensor correct_weight_update(weight_update);
//		baseline_conv2D_update(input, gradient_next, correct_weight_update);
//
//		const int workspace_size = max(math::convolution2D::explicitGemmWorkspace(input.shape(), gradient_next.shape(), weight_update.shape()));
//		Tensor workspace( { workspace_size }, "float32", Device::cpu());
//		testing::initForTest(workspace, 0.0f);
//		math::convolution2D::explicitGemmUpdate(DeviceContext(), input, gradient_next, weight_update, workspace);
//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			storage.moveTo(Device::cuda(0));
//			input.moveTo(Device::cuda(0));
//			gradient_next.moveTo(Device::cuda(0));
//			testing::initForTest(weight_update, 1.57f);
//			weight_update.moveTo(Device::cuda(0));
//			workspace.moveTo(Device::cuda(0));
//
//			math::convolution2D::explicitGemmUpdate(context, input, gradient_next, weight_update, workspace);
//			EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//		}
//	}
//
//	TEST(TestConv2D, winograd4x4_conv2D_3x3_forward)
//	{
//		DeviceContext context(Device::cpu());
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor output( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weights( { 21, 3, 3, 35 }, "float32", Device::cpu());
//		Tensor bias( { 21 }, "float32", Device::cpu());
//		testing::initForTest(weights, 0.0f);
//		testing::initForTest(input, 1.0f);
//		testing::initForTest(bias, 1.0f);
//
//		Tensor correct_output(output.shape(), "float32", Device::cpu());
//		baseline_conv2D_forward(input, correct_output, weights, bias, NonlinearityType::SIGMOID);
//
//		Tensor weight_matrices = Tensor( { 36, 21, 35 }, "float32", Device::cpu());
//		math::winograd4x4TransformWeight(context, weights, weight_matrices, false);
//
//		Tensor input_matrices = Tensor( { 36, 2 * 4 * 5, 35 }, "float32", Device::cpu());
//		Tensor output_matrices = Tensor( { 36, 2 * 4 * 5, 21 }, "float32", Device::cpu());
//		math::winograd4x4TransformInput(context, input, input_matrices);
//		math::gemmBatched(context, 'n', 't', output_matrices, input_matrices, weight_matrices);
//		math::winograd4x4TransformOutput(context, output, output_matrices, bias, nullptr, NonlinearityType::SIGMOID);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//	}
//	TEST(TestConv2D, winograd4x4_conv2D_3x3_backward)
//	{
//		DeviceContext context(Device::cpu());
//		Tensor gradient_prev( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor output( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor gradient_next(output.shape(), "float32", Device::cpu());
//		Tensor weights( { output.lastDim(), 3, 3, gradient_prev.lastDim() }, "float32", Device::cpu());
//		ml::testing::initForTest(output, 0.0f);
//		ml::testing::initForTest(gradient_next, 1.0f);
//		ml::testing::initForTest(weights, 1.57f);
//
//		Tensor correct_gradient_prev(gradient_prev);
//		baseline_conv2D_backward(output, correct_gradient_prev, gradient_next, weights, NonlinearityType::SIGMOID);
//
//		ml::testing::initForTest(gradient_next, 1.0f);
//		math::nonlinearityBackwardInPlace(context, gradient_next, output, NonlinearityType::SIGMOID);
//
//		Tensor weight_matrices = Tensor( { 36, 21, 35 }, "float32", Device::cpu());
//		Tensor gradient_prev_matrices = Tensor( { 36, 2 * 4 * 5, 35 }, "float32", Device::cpu());
//		Tensor gradient_next_matrices = Tensor( { 36, 2 * 4 * 5, 21 }, "float32", Device::cpu());
//
//		math::winograd4x4TransformWeight(context, weights, weight_matrices, true);
//		math::winograd4x4TransformInput(context, gradient_next, gradient_next_matrices);
//		math::gemmBatched(context, 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices);
//		math::winograd4x4TransformOutput(context, gradient_prev, gradient_prev_matrices, Tensor( { }, "float32", Device::cpu()));
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
//	}
//	TEST(TestConv2D, winograd4x4_conv2D_3x3_update)
//	{
//		DeviceContext context(Device::cpu());
//		Tensor input( { 2, 13, 17, 35 }, "float32", Device::cpu());
//		Tensor gradient_next( { 2, 13, 17, 21 }, "float32", Device::cpu());
//		Tensor weight_update( { 21, 3, 3, 35 }, "float32", Device::cpu());
//		Tensor storage( { 8, 21 }, "float32", Device::cpu());
//		testing::initForTest(input, 0.0f);
//		testing::initForTest(gradient_next, 1.0f);
//		testing::initForTest(weight_update, 1.57f);
//
//		Tensor correct_weight_update(weight_update);
//		baseline_conv2D_update(input, gradient_next, correct_weight_update);
//
//		Tensor weight_update_matrices = Tensor( { 36, 21, 35 }, "float32", Device::cpu());
//		Tensor gradient_prev_matrices = Tensor( { 36, 2 * 4 * 5, 35 }, "float32", Device::cpu());
//		Tensor gradient_next_matrices = Tensor( { 36, 2 * 4 * 5, 21 }, "float32", Device::cpu());
//
//		math::winograd4x4TransformGradient(context, gradient_next, gradient_next_matrices);
//		math::winograd4x4TransformInput(context, input, gradient_prev_matrices);
//		math::gemmBatched(context, 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices);
//		math::winograd4x4TransformUpdate(context, weight_update, weight_update_matrices);
//
//		EXPECT_LE(testing::diffForTest(correct_weight_update, weight_update), 1.0e-4f);
//	}

} /* namespace ml */
