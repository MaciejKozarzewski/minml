/*
 * test_quantization.cpp
 *
 *  Created on: Feb 8, 2025
 *      Author: Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/json.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace
{
	using namespace ml;

	bool is_int8(const Tensor &t) noexcept
	{
		return t.dtype() == DataType::INT8;
	}
	bool is_fp32(const Tensor &t) noexcept
	{
		return t.dtype() == DataType::FLOAT32;
	}

	int8_t quantize(float x) noexcept
	{
		return std::max(-128.0f, std::min(127.0f, std::round(x)));
	}

	float cpu_act_forward(ActivationType act, float x) noexcept
	{
		switch (act)
		{
			default:
			case ActivationType::LINEAR:
				return x;
			case ActivationType::SIGMOID:
				return 1.0f / (1.0f + std::exp(-x));
			case ActivationType::TANH:
				return std::tanh(x);
			case ActivationType::RELU:
				return std::max(0.0f, x);
			case ActivationType::GELU:
				return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
			case ActivationType::EXP:
				return std::exp(x);
		}
	}

	void baseline_int8_conv2D_forward(const Tensor &input, AffineTransform input_transform, Tensor &output, AffineTransform output_transform,
			const Tensor &weights, const Tensor &channel_scales, const Tensor &bias, const Tensor &ext, AffineTransform ext_transform,
			ActivationType act)
	{
		assert(is_int8(input));
		assert(is_int8(output) || is_fp32(output));
		assert(is_int8(weights));
		assert(is_fp32(channel_scales));
		assert(is_fp32(bias));
		assert(input.dim(3) == weights.dim(3)); // input filters
		assert(output.dim(3) == weights.dim(0)); // output filters

		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters_in = input.dim(3);
		const int filters_out = output.dim(3);

		const int kernel_height = weights.dim(1);
		const int kernel_width = weights.dim(2);
		const int pad_h = -(kernel_height - 1) / 2;
		const int pad_w = -(kernel_width - 1) / 2;

		const int32_t input_zero = get_zero<int8_t>(input_transform);
		const AffineTransform output_to_int8 = output_transform.get_inverse();

		for (int b = 0; b < batch; b++)
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
					for (int out = 0; out < filters_out; out++)
					{
						int32_t acc = 0;
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
							{
								const int x = pad_h + h + i;
								const int y = pad_w + w + j;
								if (0 <= x and x < height and 0 <= y and y < width)
								{
									for (int in = 0; in < filters_in; in++)
										acc += (int) weights.at( { out, i, j, in }) * (int) input.at( { b, x, y, in });
								}
								else
								{
									for (int in = 0; in < filters_in; in++)
										acc += (int) weights.at( { out, i, j, in }) * input_zero;
								}
							}
						// quantization shift of the input tensor is absorbed into the bias
						float tmp = static_cast<float>(acc) * (float) channel_scales.at( { out }) + (float) bias.at( { out });

						if (not ext.isEmpty())
							tmp += ext_transform((float) ext.at( { b, h, w, out }));

						tmp = cpu_act_forward(act, tmp);

						if (is_int8(output))
							output.at( { b, h, w, out }) = quantize(output_to_int8(tmp));
						if (is_fp32(output))
							output.at( { b, h, w, out }) = tmp;
					}
	}
	void baseline_int8_depthwise_conv2D_forward(const Tensor &input, AffineTransform input_transform, Tensor &output,
			AffineTransform output_transform, const Tensor &weights, const Tensor &channel_scales, const Tensor &bias)
	{
		assert(is_int8(input));
		assert(is_int8(output) || is_fp32(output));
		assert(is_int8(weights));
		assert(is_fp32(channel_scales));
		assert(is_fp32(bias));
		assert(input.device().isCPU());
		assert(input.rank() == 4);
		assert(output.rank() == 4);
		assert(weights.rank() == 3);
		assert(output.lastDim() == weights.lastDim()); // output filters

		const int batch = input.dim(0);
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int filters = input.dim(3);

		const int kernel_height = weights.dim(0);
		const int kernel_width = weights.dim(1);

		const int pad_h = -(kernel_height - 1) / 2;
		const int pad_w = -(kernel_width - 1) / 2;

		const int32_t input_zero = get_zero<int8_t>(input_transform);
		const AffineTransform output_to_int8 = output_transform.get_inverse();

		for (int b = 0; b < batch; b++)
			for (int f = 0; f < filters; f++)
				for (int h = 0; h < height; h++)
					for (int w = 0; w < width; w++)
					{
						int32_t acc = 0;
						for (int i = 0; i < kernel_height; i++)
							for (int j = 0; j < kernel_width; j++)
							{
								const int x = pad_h + h + i;
								const int y = pad_w + w + j;
								if (0 <= x and x < height and 0 <= y and y < width)
									acc += (int) weights.at( { i, j, f }) * (int) input.at( { b, x, y, f });
								else
									acc += (int) weights.at( { i, j, f }) * input_zero;
							}
						float tmp = static_cast<float>(acc) * (float) channel_scales.at( { f }) + (float) bias.at( { f });

						if (is_int8(output))
							output.at( { b, h, w, f }) = quantize(output_to_int8(tmp));
						if (is_fp32(output))
							output.at( { b, h, w, f }) = tmp;
					}
	}
}
namespace ml
{
	TEST(TestQuantization, conv2D_1x1_forward)
	{
		const int batch_size = 11;
		const int height = 12;
		const int width = 13;
		const int input_channels = 36;
		const int output_channels = 44;

		Context context(Device::cpu());
		Tensor input( { batch_size, height, width, input_channels }, "int8", Device::cpu());
		Tensor output_int8( { batch_size, height, width, output_channels }, "int8", Device::cpu());
		Tensor output_fp32(output_int8.shape(), "float32", Device::cpu());
		Tensor weights( { output_channels, 1, 1, input_channels }, "int8", Device::cpu());
		Tensor bias( { output_channels }, "float32", Device::cpu());
		Tensor channel_scales = zeros_like(bias);
		Tensor ext = zeros_like(output_int8);
		ml::testing::initForTest(weights, 0.0);
		ml::testing::initForTest(input, 1.0);
		ml::testing::initForTest(bias, 1.0);
		ml::testing::initForTest(channel_scales, 1.0, 0.001);
		ml::testing::initForTest(ext, 0);

		const AffineTransform input_transform(0.123f, 4.56f);
		const AffineTransform output_transform(0.078f, 0.9f);
		const AffineTransform ext_transform(0.078f, -0.9f);

		Tensor correct_output_int8 = zeros_like(output_int8);
		Tensor correct_output_fp32 = zeros_like(output_fp32);
		baseline_int8_conv2D_forward(input, input_transform, correct_output_int8, output_transform, weights, channel_scales, bias, ext, ext_transform,
				ActivationType::RELU);
		baseline_int8_conv2D_forward(input, input_transform, correct_output_fp32, output_transform, weights, channel_scales, bias, ext, ext_transform,
				ActivationType::RELU);

//		Tensor weight_matrices = weights.view(Shape( { output_channels, input_channels }));
//		Tensor input_matrices = input.view(Shape( { batch_size * height * width, input_channels }));
//		Tensor output_matrices = output.view(Shape( { batch_size * height * width, output_channels }));
//
//		gemm_ex(context, output_matrices, 1.0f, 'n', input_matrices, 't', weight_matrices, 0.0f, output_matrices, bias, ActivationType::SIGMOID);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::INT8))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output_int8.moveTo(device);
			output_fp32.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			channel_scales.moveTo(device);
			ext.moveTo(device);
			Tensor weight_matrices = weights.view(Shape( { output_channels, input_channels }));
			Tensor input_matrices = input.view(Shape( { batch_size * height * width, input_channels }));
			Tensor output_matrices( { batch_size * height * width, output_channels }, "int32", device);

			output_int8.zeroall();
			output_fp32.zeroall();

			gemm(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1, 0);
			quantized_scale_shift_act(context, output_int8, output_transform, output_matrices, channel_scales, bias, ActivationType::RELU, ext,
					ext_transform);
			quantized_scale_shift_act(context, output_fp32, output_transform, output_matrices, channel_scales, bias, ActivationType::RELU, ext,
					ext_transform);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output_int8, output_int8), 1.0e-4f);
			EXPECT_LE(ml::testing::diffForTest(correct_output_fp32, output_fp32), 1.0e-4f);
		}
	}
	TEST(TestQuantization, conv2D_5x5_forward)
	{
		const int batch_size = 3;
		const int height = 12;
		const int width = 13;
		const int input_channels = 36;
		const int output_channels = 44;

		Context context(Device::cpu());
		Tensor input( { batch_size, height, width, input_channels }, "int8", Device::cpu());
		Tensor output_int8( { batch_size, height, width, output_channels }, "int8", Device::cpu());
		Tensor output_fp32(output_int8.shape(), "float32", Device::cpu());
		Tensor weights( { output_channels, 5, 5, input_channels }, "int8", Device::cpu());
		Tensor bias( { output_channels }, "float32", Device::cpu());
		Tensor channel_scales = zeros_like(bias);
		Tensor ext = zeros_like(output_int8);
		ml::testing::initForTest(weights, 0.0);
		ml::testing::initForTest(input, 1.0);
		ml::testing::initForTest(bias, 1.0);
		ml::testing::initForTest(channel_scales, 1.0, 0.0001);
		ml::testing::initForTest(ext, 0);

		const AffineTransform input_transform(0.123f, 4.56f);
		const AffineTransform output_transform(0.078f, 0.9f);
		const AffineTransform ext_transform(0.078f, -0.9f);

		const int32_t input_zero = get_zero<int8_t>(input_transform);

		Tensor correct_output_int8 = zeros_like(output_int8);
		Tensor correct_output_fp32 = zeros_like(output_fp32);
		baseline_int8_conv2D_forward(input, input_transform, correct_output_int8, output_transform, weights, channel_scales, bias, ext, ext_transform,
				ActivationType::RELU);
		baseline_int8_conv2D_forward(input, input_transform, correct_output_fp32, output_transform, weights, channel_scales, bias, ext, ext_transform,
				ActivationType::RELU);

//		Tensor weight_matrices = weights.view(Shape( { output_channels, input_channels }));
//		Tensor input_matrices = input.view(Shape( { batch_size * height * width, input_channels }));
//		Tensor output_matrices = output.view(Shape( { batch_size * height * width, output_channels }));
//
//		gemm_ex(context, output_matrices, 1.0f, 'n', input_matrices, 't', weight_matrices, 0.0f, output_matrices, bias, ActivationType::SIGMOID);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::INT8))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output_int8.moveTo(device);
			output_fp32.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			channel_scales.moveTo(device);
			ext.moveTo(device);
			Tensor weight_matrices = weights.view(Shape( { output_channels, 5 * 5 * input_channels }));
			Tensor input_matrices( { batch_size * height * width, 5 * 5 * input_channels }, "int8", device);
			Tensor output_matrices( { batch_size * height * width, output_channels }, "int32", device);

			output_int8.zeroall();
			output_fp32.zeroall();

			create_receptive_fields(context, input_matrices, input, 5, &input_zero);
			gemm(context, 'n', 't', output_matrices, input_matrices, weight_matrices, 1, 0);
			quantized_scale_shift_act(context, output_int8, output_transform, output_matrices, channel_scales, bias, ActivationType::RELU, ext,
					ext_transform);
			quantized_scale_shift_act(context, output_fp32, output_transform, output_matrices, channel_scales, bias, ActivationType::RELU, ext,
					ext_transform);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output_int8, output_int8), 1.0e-4f);
			EXPECT_LE(ml::testing::diffForTest(correct_output_fp32, output_fp32), 1.0e-4f);
		}
	}
	TEST(TestQuantization, depthwise_conv2D_forward)
	{
		const int batch_size = 1;
		const int height = 1;
		const int width = 1;
		const int channels = 36;
		const int kernel = 7;

		Context context(Device::cpu());
		Tensor input( { batch_size, height, width, channels }, "int8", Device::cpu());
		Tensor output(input.shape(), "int8", Device::cpu());
		Tensor weights( { kernel, kernel, channels }, "int8", Device::cpu());
		Tensor bias( { channels }, "float32", Device::cpu());
		Tensor channel_scales = zeros_like(bias);
		ml::testing::initForTest(weights, 0.0);
		ml::testing::initForTest(input, 1.0);
		ml::testing::initForTest(bias, 1.0);
		ml::testing::initForTest(channel_scales, 1.0, 0.001);

		const AffineTransform input_transform(0.123f, 4.56f);
		const AffineTransform output_transform(0.078f, 0.9f);
		const int32_t input_zero = get_zero<int8_t>(input_transform);

		Tensor correct_output = zeros_like(output);
		baseline_int8_depthwise_conv2D_forward(input, input_transform, correct_output, output_transform, weights, channel_scales, bias);

//		Tensor weight_matrices = weights.view(Shape( { output_channels, input_channels }));
//		Tensor input_matrices = input.view(Shape( { batch_size * height * width, input_channels }));
//		Tensor output_matrices = output.view(Shape( { batch_size * height * width, output_channels }));
//
//		gemm_ex(context, output_matrices, 1.0f, 'n', input_matrices, 't', weight_matrices, 0.0f, output_matrices, bias, ActivationType::SIGMOID);
//		EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);

		if (ml::testing::has_device_supporting(DataType::INT8))
		{
			const Device device = ml::testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			bias.moveTo(device);
			channel_scales.moveTo(device);

			output.zeroall();

			quantized_depthwise_conv_forward(context, input, weights, channel_scales, bias, output, output_transform, input_zero);
			context.synchronize();
			EXPECT_LE(ml::testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
}
