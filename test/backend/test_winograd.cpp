/*
 * test_winograd.cpp
 *
 *  Created on: Dec 20, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/testing_util.hpp>

#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	void print_tile(const Tensor &tile)
	{
		std::cout << "---------------------------------------" << std::endl;
		for (int i = 0; i < tile.dim(0); i++)
		{
			for (int j = 0; j < tile.dim(1); j++)
				std::cout << tile.get( { i, j }) << " ";
			std::cout << std::endl;
		}
		std::cout << "---------------------------------------" << std::endl;
	}
	void extract_input_tile(Tensor &tile, const Tensor &matrices, int tile_idx, int filter)
	{
		assert(tile.device().isCPU());
		for (int i = 0; i < tile.dim(0); i++)
			for (int j = 0; j < tile.dim(1); j++)
				tile.set(matrices.get( { i * tile.dim(1) + j, tile_idx, filter }), { i, j });
	}
	void extract_output_tile(Tensor &tile, const Tensor &tensor, int b, int h, int w, int filter)
	{
		assert(tile.device().isCPU());
		for (int i = 0; i < tile.dim(0); i++)
			for (int j = 0; j < tile.dim(1); j++)
				if ((h + i) < tensor.dim(1) && (w + j) < tensor.dim(2))
					tile.set(tensor.get( { b, h + i, w + j, filter }), { i, j });
	}
	void extract_weight_tile(Tensor &tile, const Tensor &matrices, int out, int in)
	{
		assert(tile.device().isCPU());
		for (int i = 0; i < tile.dim(0); i++)
			for (int j = 0; j < tile.dim(1); j++)
				tile.set(matrices.get( { i * tile.dim(1) + j, out, in }), { i, j });
	}
	void extract_update_tile(Tensor &tile, const Tensor &update, int out, int in)
	{
		assert(tile.device().isCPU());
		for (int i = 0; i < tile.dim(0); i++)
			for (int j = 0; j < tile.dim(1); j++)
				tile.set(update.get( { out, i, j, in }), { i, j });
	}

	void winograd_input_transform(Tensor &matrices, const Tensor &input, const Tensor &transform_matrix, int winograd_tile)
	{
		assert(same_device(matrices, input, transform_matrix));
		assert(input.device().isCPU());
		const int input_tile_size = winograd_tile + 3 - 1;
		Tensor tile( { input_tile_size, input_tile_size }, DataType::FLOAT32, Device::cpu());
		Tensor tmp(tile.shape(), tile.dtype(), tile.device());

		int tile_index = 0;
		for (int b = 0; b < input.dim(0); b++) // loop over batches
			for (int h = 0; h < input.dim(1); h += winograd_tile) // loop over height of image
				for (int w = 0; w < input.dim(2); w += winograd_tile) // loop over width of image
				{
					for (int f = 0; f < input.dim(3); f++) // loop over filters
					{
						for (int i = -1; i < input_tile_size - 1; i++)
							for (int j = -1; j < input_tile_size - 1; j++)
								if ((h + i) >= 0 && (h + i) < input.dim(1) && (w + j) >= 0 && (w + j) < input.dim(2))
									tile.set(input.get( { b, h + i, w + j, f }), { 1 + i, 1 + j });
								else
									tile.set(0.0f, { 1 + i, 1 + j });

						gemm(Context(), 'n', 'n', tmp, transform_matrix, tile, 1, 0);
						gemm(Context(), 'n', 't', tile, tmp, transform_matrix, 1, 0);

						for (int i = 0; i < tile.dim(0); i++)
							for (int j = 0; j < tile.dim(1); j++)
								matrices.set(tile.get( { i, j }), { i * input_tile_size + j, tile_index, f });
					}
					tile_index++;
				}
	}
	void winograd_output_transform(Tensor &output, const Tensor &matrices, const Tensor &transform_matrix, int winograd_tile, const Tensor *bias,
			const Tensor *add, bool use_relu)
	{
		assert(same_device(output, matrices, transform_matrix));
		assert(output.device().isCPU());
		const int input_tile_size = winograd_tile + 3 - 1;
		Tensor tile( { input_tile_size, input_tile_size }, DataType::FLOAT32, Device::cpu());
		Tensor tmp1( { winograd_tile, input_tile_size }, tile.dtype(), tile.device());
		Tensor tmp2( { winograd_tile, winograd_tile }, tile.dtype(), tile.device());

		int tile_index = 0;
		for (int b = 0; b < output.dim(0); b++) // loop over batches
			for (int h = 0; h < output.dim(1); h += winograd_tile) // loop over height of image
				for (int w = 0; w < output.dim(2); w += winograd_tile) // loop over width of image
				{
					for (int f = 0; f < output.dim(3); f++) // loop over filters
					{
						for (int i = 0; i < tile.dim(0); i++)
							for (int j = 0; j < tile.dim(1); j++)
								tile.set(matrices.get( { i * tile.dim(1) + j, tile_index, f }), { i, j });

						gemm(Context(), 'n', 'n', tmp1, transform_matrix, tile, 1, 0);
						gemm(Context(), 'n', 't', tmp2, tmp1, transform_matrix, 1, 0);

						for (int i = 0; i < tmp2.dim(0); i++)
							for (int j = 0; j < tmp2.dim(1); j++)
								if ((h + i) < output.dim(1) && (w + j) < output.dim(2))
								{
									float tmp = tmp2.get( { i, j });
									if (bias != nullptr)
										tmp += bias->get( { f });
									if (add != nullptr)
										tmp += add->get( { b, h + i, w + j, f });
									if (use_relu)
										tmp = std::max(0.0f, tmp);
									output.set(tmp, { b, h + i, w + j, f });
								}
					}
					tile_index++;
				}
	}
	void winograd_weight_transform(Tensor &matrices, const Tensor &weight, const Tensor &transform_matrix, int winograd_tile)
	{
		assert(same_device(matrices, weight, transform_matrix));
		assert(matrices.device().isCPU());
		Tensor kernel( { 3, 3 }, DataType::FLOAT32, Device::cpu());
		Tensor tmp1( { winograd_tile + 3 - 1, 3 }, DataType::FLOAT32, Device::cpu());
		Tensor tmp2( { winograd_tile + 3 - 1, winograd_tile + 3 - 1 }, DataType::FLOAT32, Device::cpu());

		for (int out = 0; out < weight.dim(0); out++) // loop over output filters
			for (int in = 0; in < weight.dim(3); in++) // loop over input filters
			{
				for (int i = 0; i < kernel.dim(0); i++)
					for (int j = 0; j < kernel.dim(1); j++)
						kernel.set(weight.get( { out, i, j, in }), { i, j });

				gemm(Context(), 'n', 'n', tmp1, transform_matrix, kernel, 1, 0);
				gemm(Context(), 'n', 't', tmp2, tmp1, transform_matrix, 1, 0);

				for (int i = 0; i < tmp2.dim(0); i++)
					for (int j = 0; j < tmp2.dim(1); j++)
						matrices.set(tmp2.get( { i, j }), { i * tmp2.dim(1) + j, out, in });
			}
	}
	void winograd_gradient_transform(const Tensor &gradient, Tensor &matrices, const Tensor &transform_matrix, int winograd_tile)
	{
		assert(same_device(matrices, gradient, transform_matrix));
		assert(matrices.device().isCPU());
		const int input_tile_size = winograd_tile + 3 - 1;
		Tensor tile( { winograd_tile, winograd_tile }, DataType::FLOAT32, Device::cpu());
		Tensor tmp1( { input_tile_size, winograd_tile }, tile.dtype(), tile.device());
		Tensor tmp2( { input_tile_size, input_tile_size }, tile.dtype(), tile.device());

		int tile_index = 0;
		for (int b = 0; b < gradient.dim(0); b++) // loop over batches
			for (int h = 0; h < gradient.dim(1); h += winograd_tile) // loop over height of image
				for (int w = 0; w < gradient.dim(2); w += winograd_tile) // loop over width of image
				{
					for (int f = 0; f < gradient.dim(3); f++) // loop over filters
					{
						for (int i = 0; i < tile.dim(0); i++)
							for (int j = 0; j < tile.dim(1); j++)
								if ((h + i) < gradient.dim(1) && (w + j) < gradient.dim(2))
									tile.set(gradient.get( { b, h + i, w + j, f }), { i, j });
								else
									tile.set(0.0f, { i, j });

						gemm(Context(), 'n', 'n', tmp1, transform_matrix, tile, 1, 0);
						gemm(Context(), 'n', 't', tmp2, tmp1, transform_matrix, 1, 0);

						for (int i = 0; i < tmp2.dim(0); i++)
							for (int j = 0; j < tmp2.dim(1); j++)
								matrices.set(tmp2.get( { i, j }), { i * input_tile_size + j, tile_index, f });
					}
					tile_index++;
				}
	}
	void winograd_update_transform(const Tensor &matrices, Tensor &weight, const Tensor &transform_matrix, int winograd_tile)
	{
		assert(same_device(matrices, weight, transform_matrix));
		assert(matrices.device().isCPU());
		Tensor tmp1( { winograd_tile + 2, winograd_tile + 2 }, DataType::FLOAT32, Device::cpu());
		Tensor tmp2( { 3, winograd_tile + 2 }, DataType::FLOAT32, Device::cpu());
		Tensor kernel( { 3, 3 }, DataType::FLOAT32, Device::cpu());

		for (int out = 0; out < weight.dim(0); out++) // loop over output filters
			for (int in = 0; in < weight.dim(3); in++) // loop over input filters
			{
				for (int i = 0; i < tmp1.dim(0); i++)
					for (int j = 0; j < tmp1.dim(1); j++)
						tmp1.set(matrices.get( { i * tmp1.dim(1) + j, out, in }), { i, j });

				gemm(Context(), 'n', 'n', tmp2, transform_matrix, tmp1, 1, 0);
				gemm(Context(), 'n', 't', kernel, tmp2, transform_matrix, 1, 0);

				for (int i = 0; i < kernel.dim(0); i++)
					for (int j = 0; j < kernel.dim(1); j++)
						weight.set(kernel.get( { i, j }), { out, i, j, in });
			}
	}
}

namespace ml
{
	TEST(TestWinograd3x3_4x4, inputTransformBaseline)
	{
		Tensor matrices( { 36, 1, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor input( { 1, 4, 4, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor transform_matrix = toTensor( { { 1.0f, 0.0f, -1.25f, 0.0f, 0.25f, 0.0f }, { 0.0f, 1.0f, 1.0f, -0.25f, -0.25f, 0.0f }, { 0.0f, -1.0f,
				1.0f, 0.25f, -0.25f, 0.0f }, { 0.0f, -1.0f, -0.5f, 1.0f, 0.5f, 0.0f }, { 0.0f, 1.0f, -0.5f, -1.0f, 0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f,
				-1.25f, 0.0f, 0.25f } });

		input.setall(Context(), 1.0f);
		winograd_input_transform(matrices, input, transform_matrix, 4);
		Tensor correct1 = toTensor( { { 1.0f, -1.5f, 0.0f, 0.0f, 0.0f, 0.25f }, { -1.5f, 2.25f, 0.0f, 0.0f, 0.0f, -0.375f }, { 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.25f, -0.375f, 0.0f, 0.0f, 0.0f,
				0.0625f } });

		Tensor tile( { 6, 6 }, DataType::FLOAT32, Device::cpu());
		for (int i = 0; i < matrices.dim(1); i++)
			for (int j = 0; j < matrices.dim(2); j++)
			{
				extract_input_tile(tile, matrices, i, j);
				EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);
			}

		for (int i = 0; i < 16; i++)
		{
			reinterpret_cast<float*>(input.data())[2 * i] = static_cast<float>(1 + i);
			reinterpret_cast<float*>(input.data())[2 * i + 1] = static_cast<float>(1 + i);
		}
		winograd_input_transform(matrices, input, transform_matrix, 4);
		Tensor correct2 = toTensor( { { 3.5f, -4.25f, -0.75f, -3.0f, 1.0f, 3.25f }, { -1.25f, 0.375f, 1.125f, 4.5f, -1.5f, -3.875f }, { -3.0f, 4.5f,
				0.0f, 0.0f, 0.0f, -0.75f }, { -12.0f, 18.0f, 0.0f, 0.0f, 0.0f, -3.0f }, { 4.0f, -6.0f, 0.0f, 0.0f, 0.0f, 1.0f }, { 10.375f, -15.3125f,
				-0.1875f, -0.75f, 0.25f, 3.1875f } });
		extract_input_tile(tile, matrices, 0, 0);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
		extract_input_tile(tile, matrices, 0, 1);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, outputTransformBaseline)
	{
		Tensor input( { 36, 1, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor output( { 1, 4, 4, 2 }, DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor( { { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f }, { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 1.0f,
				1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 2.0f } });

		input.setall(Context(), 1.0f);
		winograd_output_transform(output, input, transform_matrix, 4, nullptr, nullptr, false);
		Tensor correct1 = toTensor( { { 12.25f, 0.0f, 14.0f, 7.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 14.0f, 0.0f, 16.0f, 8.0f }, { 7.0f, 0.0f, 8.0f,
				4.0f } });

		Tensor tile( { 4, 4 }, DataType::FLOAT32, Device::cpu());

		extract_output_tile(tile, output, 0, 0, 0, 0);
		EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);
		extract_output_tile(tile, output, 0, 0, 0, 1);
		EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);

		for (int i = 0; i < 36; i++)
		{
			reinterpret_cast<float*>(input.data())[2 * i] = static_cast<float>(1 + i);
			reinterpret_cast<float*>(input.data())[2 * i + 1] = static_cast<float>(1 + i);
		}
		winograd_output_transform(output, input, transform_matrix, 4, nullptr, nullptr, false);
		Tensor correct2 = toTensor( { { 128.625f, -5.25f, 163.0f, 88.5f }, { -31.5f, 0.0f, -36.0f, -18.0f }, { 243.0f, -6.0f, 296.0f, 156.0f }, {
				163.5f, -3.0f, 196.0f, 102.0f } });
		extract_output_tile(tile, output, 0, 0, 0, 0);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
		extract_output_tile(tile, output, 0, 0, 0, 1);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, weightTransformBaseline)
	{
		Tensor matrices( { 36, 1, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor weight( { 1, 3, 3, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor transform_matrix = toTensor(
				{ { 1.0f, 0.0f, 0.0f }, { 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f }, { 2.0f / 3.0f, -2.0f / 3.0f, 2.0f / 3.0f }, { 1.0f / 3.0f, 2.0f
						/ 3.0f, 4.0f / 3.0f }, { 1.0f / 3.0f, -2.0f / 3.0f, 4.0f / 3.0f }, { 0.0f, 0.0f, 2.0f } });

		weight.setall(Context(), 9.0f);
		winograd_weight_transform(matrices, weight, transform_matrix, 4);
		Tensor correct1 = toTensor( { { 9.0f, 18.0f, 6.0f, 21.0f, 9.0f, 18.0f }, { 18.0f, 36.0f, 12.0f, 42.0f, 18.0f, 36.0f }, { 6.0f, 12.0f, 4.0f,
				14.0f, 6.0f, 12.0f }, { 21.0f, 42.0f, 14.0f, 49.0f, 21.0f, 42.0f }, { 9.0f, 18.0f, 6.0f, 21.0f, 9.0f, 18.0f }, { 18.0f, 36.0f, 12.0f,
				42.0f, 18.0f, 36.0f } });

		Tensor tile( { 6, 6 }, DataType::FLOAT32, Device::cpu());
		for (int i = 0; i < matrices.dim(1); i++)
			for (int j = 0; j < matrices.dim(2); j++)
			{
				extract_weight_tile(tile, matrices, i, j);
				EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);
			}

		for (int i = 0; i < 9; i++)
		{
			reinterpret_cast<float*>(weight.data())[2 * i] = static_cast<float>(1 + i);
			reinterpret_cast<float*>(weight.data())[2 * i + 1] = static_cast<float>(1 + i);
		}
		winograd_weight_transform(matrices, weight, transform_matrix, 4);
		Tensor correct2 = toTensor( { { 1.0f, 4.0f, 1.3333333333f, 5.6666666667f, 3.0f, 6.0f }, { 8.0f, 20.0f, 6.6666666667f, 25.3333333333f, 12.0f,
				24.0f }, { 2.6666666667f, 6.6666666667f, 2.222222222f, 8.4444444444f, 4.0f, 8.0f }, { 12.3333333333f, 29.3333333333f, 9.7777777778f,
				36.5555555556f, 17.0f, 34.0f }, { 7.0f, 16.0f, 5.3333333333f, 19.6666666667f, 9.0f, 18.0f }, { 14.0f, 32.0f, 10.6666666667f,
				39.3333333333f, 18.0f, 36.0f } });
		extract_weight_tile(tile, matrices, 0, 0);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
		extract_weight_tile(tile, matrices, 0, 1);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, gradientTransformBaseline)
	{
		Tensor gradient( { 1, 4, 4, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices( { 36, 1, 2 }, DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor(
				{ { 1.0f, 0.0f, 0.0f, 0.0f }, { 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f }, { 2.0f / 3.0f, -2.0f / 3.0f, 2.0f / 3.0f, -2.0f
						/ 3.0f }, { 1.0f / 3.0f, 2.0f / 3.0f, 4.0f / 3.0f, 8.0f / 3.0f }, { 1.0f / 3.0f, -2.0f / 3.0f, 4.0f / 3.0f, -8.0f / 3.0f }, {
						0.0f, 0.0f, 0.0f, 2.0f } });

		gradient.setall(Context(), 1.0f);
		winograd_gradient_transform(gradient, matrices, transform_matrix, 4);
		Tensor correct1 = toTensor( { { 1.0f, 2.66666667f, 0.0f, 5.0f, -1.66666667f, 2.0f }, { 2.66666667f, 7.11111111, 0.0f, 13.33333333f,
				-4.44444444, 5.33333333f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 5.0f, 13.33333333f, 0.0f, 25.0f, -8.33333333f, 10.0f }, {
				-1.66666667f, -4.44444444, 0.0f, -8.33333333f, 2.77777778f, -3.33333333f }, { 2.0f, 5.33333333f, 0.0f, 10.0f, -3.33333333f, 4.0f } });

		Tensor tile( { 6, 6 }, DataType::FLOAT32, Device::cpu());

		extract_input_tile(tile, matrices, 0, 0);
		EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);
		extract_input_tile(tile, matrices, 0, 1);
		EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);

		for (int i = 0; i < 16; i++)
		{
			reinterpret_cast<float*>(gradient.data())[2 * i] = static_cast<float>(1 + i);
			reinterpret_cast<float*>(gradient.data())[2 * i + 1] = static_cast<float>(1 + i);
		}
		winograd_gradient_transform(gradient, matrices, transform_matrix, 4);
		Tensor correct2 = toTensor( { { 1.0f, 6.66666667f, -1.33333333f, 16.33333333f, -7.66666667f, 8.0f }, { 18.66666667f, 60.44444444f,
				-3.55555555f, 123.55555555f, -47.11111111f, 53.33333333f }, { -5.33333333f, -14.22222222f, 0.0f, -26.66666667f, 8.88888889f,
				-10.66666667f }, { 50.33333333f, 154.22222222f, -6.66666667f, 308.33333333f, -113.88888889f, 130.66666667f }, { -25.66666667f,
				-75.11111111, 2.22222222f, -147.22222222f, 52.77777778f, -61.33333333f }, { 26.0f, 77.33333333f, -2.66666667f, 152.66666667f,
				-55.33333333f, 64.0f } });

		extract_input_tile(tile, matrices, 0, 0);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
		extract_input_tile(tile, matrices, 0, 1);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, updateTransformBaseline)
	{
		Tensor matrices( { 36, 1, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor update( { 1, 3, 3, 2 }, DataType::FLOAT32, Device::cpu());
		Tensor transform_matrix = toTensor( { { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f, }, { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f, }, { 0.0f, 1.0f,
				1.0f, 1.0f, 1.0f, 2.0f, } });

		matrices.setall(Context(), 1.0f);
		winograd_update_transform(matrices, update, transform_matrix, 4);
		Tensor correct1 = toTensor( { { 12.25f, 0.0f, 21.0f }, { 0.0f, 0.0f, 0.0f }, { 21.0f, 0.0, 36.0f } });

		Tensor tile( { 3, 3 }, DataType::FLOAT32, Device::cpu());
		for (int i = 0; i < update.dim(0); i++)
			for (int j = 0; j < update.dim(3); j++)
			{
				extract_update_tile(tile, update, i, j);
				EXPECT_LE(testing::diffForTest(correct1, tile), 1.0e-4f);
			}

		for (int i = 0; i < 36; i++)
		{
			reinterpret_cast<float*>(matrices.data())[2 * i] = static_cast<float>(1 + i);
			reinterpret_cast<float*>(matrices.data())[2 * i + 1] = static_cast<float>(1 + i);
		}
		winograd_update_transform(matrices, update, transform_matrix, 4);
		Tensor correct2 = toTensor( { { 128.625f, -5.25f, 262.0f }, { -31.5f, 0.0f, -54.0f }, { 469.5f, -9.0f, 876.0f } });
		extract_update_tile(tile, update, 0, 0);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
		extract_update_tile(tile, update, 0, 1);
		EXPECT_LE(testing::diffForTest(correct2, tile), 1.0e-4f);
	}

	TEST(TestWinograd3x3_4x4, cpuInputTransform)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor input( { 3, 9, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		int nb_of_tiles = input.dim(0) * ((input.dim(1) + 3) / 4) * ((input.dim(2) + 3) / 4);
		Tensor matrices( { 36, nb_of_tiles, input.dim(3) }, DataType::FLOAT32, Device::cpu());
		Tensor correct(matrices.shape(), DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor( { { 1.0f, 0.0f, -1.25f, 0.0f, 0.25f, 0.0f }, { 0.0f, 1.0f, 1.0f, -0.25f, -0.25f, 0.0f }, { 0.0f, -1.0f,
				1.0f, 0.25f, -0.25f, 0.0f }, { 0.0f, -1.0f, -0.5f, 1.0f, 0.5f, 0.0f }, { 0.0f, 1.0f, -0.5f, -1.0f, 0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f,
				-1.25f, 0.0f, 0.25f } });
		winograd_input_transform(correct, input, transform_matrix, 4);

		Device::cpu().setNumberOfThreads(1);
		winogradInputTransform(Context(), weights_shape, input, matrices);

		EXPECT_LE(testing::diffForTest(correct, matrices), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cpuOutputTransform)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor output( { 3, 9, 11, 35 }, DataType::FLOAT32, Device::cpu());

		int nb_of_tiles = output.dim(0) * ((output.dim(1) + 3) / 4) * ((output.dim(2) + 3) / 4);
		Tensor matrices( { 36, nb_of_tiles, output.dim(3) }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);
		Tensor correct(output.shape(), DataType::FLOAT32, Device::cpu());
		Tensor bias;

		Tensor transform_matrix = toTensor( { { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f }, { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 1.0f,
				1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 2.0f } });
		winograd_output_transform(correct, matrices, transform_matrix, 4, nullptr, nullptr, false);

		Device::cpu().setNumberOfThreads(2);
		winogradOutputTransform(Context(), weights_shape, matrices, output, bias, Tensor(), ActivationType::LINEAR);

		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cpuOutputTransformExtension)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor output( { 3, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());

		int nb_of_tiles = output.dim(0) * ((output.dim(1) + 3) / 4) * ((output.dim(2) + 3) / 4);
		Tensor matrices( { 36, nb_of_tiles, output.dim(3) }, DataType::FLOAT32, Device::cpu());
		Tensor bias( { output.dim(3) }, DataType::FLOAT32, Device::cpu());
		Tensor add(output.shape(), DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);
		testing::initForTest(bias, 0.0f);
		Tensor correct(output.shape(), DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor( { { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f }, { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 1.0f,
				1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 2.0f } });
		winograd_output_transform(correct, matrices, transform_matrix, 4, &bias, &add, true);

		Device::cpu().setNumberOfThreads(1);
		winogradOutputTransform(Context(), weights_shape, matrices, output, bias, add, ActivationType::RELU);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);

		Device::cpu().setNumberOfThreads(2);
		winogradOutputTransform(Context(), weights_shape, matrices, output, bias, add, ActivationType::RELU);
		EXPECT_LE(testing::diffForTest(correct, output), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cpuWeightTransform)
	{
		Tensor weight( { 7, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);
		Tensor matrices( { 36, weight.dim(0), weight.dim(3) }, DataType::FLOAT32, Device::cpu());
		Tensor correct(matrices.shape(), DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor(
				{ { 1.0f, 0.0f, 0.0f }, { 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f }, { 2.0f / 3.0f, -2.0f / 3.0f, 2.0f / 3.0f }, { 1.0f / 3.0f, 2.0f
						/ 3.0f, 4.0f / 3.0f }, { 1.0f / 3.0f, -2.0f / 3.0f, 4.0f / 3.0f }, { 0.0f, 0.0f, 2.0f } });
		winograd_weight_transform(correct, weight, transform_matrix, 4);

		Device::cpu().setNumberOfThreads(2);
		winogradWeightTransform(Context(), weight, matrices, false, false);

		EXPECT_LE(testing::diffForTest(correct, matrices), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cpuGradientTransform)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor gradient( { 3, 9, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(gradient, 0.0f);

		Tensor matrices( { 36, 27, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor correct(matrices.shape(), DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor(
				{ { 1.0f, 0.0f, 0.0f, 0.0f }, { 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f }, { 2.0f / 3.0f, -2.0f / 3.0f, 2.0f / 3.0f, -2.0f
						/ 3.0f }, { 1.0f / 3.0f, 2.0f / 3.0f, 4.0f / 3.0f, 8.0f / 3.0f }, { 1.0f / 3.0f, -2.0f / 3.0f, 4.0f / 3.0f, -8.0f / 3.0f }, {
						0.0f, 0.0f, 0.0f, 2.0f } });
		winograd_gradient_transform(gradient, correct, transform_matrix, 4);

		Device::cpu().setNumberOfThreads(1);
		winogradGradientTransform(Context(), weights_shape, gradient, matrices);

		EXPECT_LE(testing::diffForTest(correct, matrices), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cpuUpdateTransform)
	{
		Tensor matrices( { 36, 7, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor update( { 7, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor correct(update.shape(), DataType::FLOAT32, Device::cpu());

		Tensor transform_matrix = toTensor( { { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f, }, { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f, }, { 0.0f, 1.0f,
				1.0f, 1.0f, 1.0f, 2.0f, } });
		winograd_update_transform(matrices, correct, transform_matrix, 4);

		winogradUpdateTransform(Context(), matrices, update);

		EXPECT_LE(testing::diffForTest(correct, update), 1.0e-4f);
	}

	TEST(TestWinograd3x3_4x4, cudaInputTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();

		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_cpu( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		winogradInputTransform(Context(), weights_shape, input, matrices_cpu);

		input.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradInputTransform(context, weights_shape, input, matrices_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cudaOutputTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		const Shape weights_shape( { 35, 3, 3, 1 });

		Tensor matrices( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor output_cpu( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		winogradOutputTransform(Context(), weights_shape, matrices, output_cpu, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor output_gpu(output_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradOutputTransform(context, weights_shape, matrices, output_gpu, Tensor(), Tensor(), ActivationType::LINEAR);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(output_cpu, output_gpu), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cudaWeightTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		Tensor weight( { 31, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);

		Tensor matrices_cpu( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		winogradWeightTransform(Context(), weight, matrices_cpu, false, false);

		weight.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradWeightTransform(context, weight, matrices_gpu, false, false);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cudaGradientTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_cpu( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		winogradGradientTransform(Context(), weights_shape, input, matrices_cpu);

		input.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradGradientTransform(context, weights_shape, input, matrices_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd3x3_4x4, cudaUpdateTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();

		Tensor matrices( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor weight_cpu( { 31, 5, 5, 35 }, DataType::FLOAT32, Device::cpu());
		winogradUpdateTransform(Context(), matrices, weight_cpu);

		matrices.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor weight_gpu(weight_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradUpdateTransform(context, matrices, weight_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(weight_cpu, weight_gpu), 1.0e-4f);
	}

	TEST(TestWinograd5x5_2x2, cudaInputTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();

		const Shape weights_shape( { 35, 5, 5, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_cpu( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cpu());
		winogradInputTransform(Context(), weights_shape, input, matrices_cpu);

		input.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradInputTransform(context, weights_shape, input, matrices_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd5x5_2x2, cudaOutputTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		const Shape weights_shape( { 35, 5, 5, 1 });

		Tensor matrices( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor output_cpu( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		winogradOutputTransform(Context(), weights_shape, matrices, output_cpu, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor output_gpu(output_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradOutputTransform(context, weights_shape, matrices, output_gpu, Tensor(), Tensor(), ActivationType::LINEAR);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(output_cpu, output_gpu), 1.0e-4f);
	}
	TEST(TestWinograd5x5_2x2, cudaWeightTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		Tensor weight( { 31, 5, 5, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);

		Tensor matrices_cpu( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		winogradWeightTransform(Context(), weight, matrices_cpu, false, false);

		weight.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradWeightTransform(context, weight, matrices_gpu, false, false);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd5x5_2x2, cudaGradientTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		const Shape weights_shape( { 35, 5, 5, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_cpu( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cpu());
		winogradGradientTransform(Context(), weights_shape, input, matrices_cpu);

		input.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor matrices_gpu(matrices_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradGradientTransform(context, weights_shape, input, matrices_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(matrices_cpu, matrices_gpu), 1.0e-4f);
	}
	TEST(TestWinograd5x5_2x2, cudaUpdateTransform)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();

		Tensor matrices( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor weight_cpu( { 31, 5, 5, 35 }, DataType::FLOAT32, Device::cpu());
		winogradUpdateTransform(Context(), matrices, weight_cpu);

		matrices.moveTo(Device::cuda(0));
		Context context(Device::cuda(0));
		Tensor weight_gpu(weight_cpu.shape(), DataType::FLOAT32, Device::cuda(0));
		winogradUpdateTransform(context, matrices, weight_gpu);
		context.synchronize();

		EXPECT_LE(testing::diffForTest(weight_cpu, weight_gpu), 1.0e-4f);
	}

	TEST(TestWinograd3x3_4x4, cpuInputTransform_bf16)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_fp32( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_bf16(matrices_fp32.shape(), DataType::BFLOAT16, Device::cpu());
		winogradInputTransform(Context(), weights_shape, input, matrices_fp32);

		input.convertTo(Context(), DataType::BFLOAT16);
		winogradInputTransform(Context(), weights_shape, input, matrices_bf16);

		matrices_bf16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_bf16), 1.0e-2f);
	}
	TEST(TestWinograd3x3_4x4, cpuOutputTransform_bf16)
	{
		const Shape weights_shape( { 35, 3, 3, 1 });

		Tensor matrices( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor output_fp32( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor output_bf16(output_fp32.shape(), DataType::BFLOAT16, Device::cpu());
		winogradOutputTransform(Context(), weights_shape, matrices, output_fp32, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.convertTo(Context(), DataType::BFLOAT16);
		winogradOutputTransform(Context(), weights_shape, matrices, output_bf16, Tensor(), Tensor(), ActivationType::LINEAR);

		output_bf16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(output_fp32, output_bf16), 1.0e-2f);
	}
	TEST(TestWinograd3x3_4x4, cpuWeightTransform_bf16)
	{
		Tensor weight( { 31, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);

		Tensor matrices_fp32( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_bf16(matrices_fp32.shape(), DataType::BFLOAT16, Device::cpu());
		winogradWeightTransform(Context(), weight, matrices_fp32, false, false);

		weight.convertTo(Context(), DataType::BFLOAT16);
		winogradWeightTransform(Context(), weight, matrices_bf16, false, false);

		matrices_bf16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_bf16), 1.0e-2f);
	}

	TEST(TestWinograd3x3_4x4, cpuInputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		const Shape weights_shape( { 35, 3, 3, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_fp32( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradInputTransform(Context(), weights_shape, input, matrices_fp32);

		input.convertTo(Context(), DataType::FLOAT16);
		winogradInputTransform(Context(), weights_shape, input, matrices_fp16);

		matrices_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}
	TEST(TestWinograd3x3_4x4, cpuOutputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		const Shape weights_shape( { 35, 3, 3, 1 });

		Tensor matrices( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor output_fp32( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor output_fp16(output_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradOutputTransform(Context(), weights_shape, matrices, output_fp32, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.convertTo(Context(), DataType::FLOAT16);
		winogradOutputTransform(Context(), weights_shape, matrices, output_fp16, Tensor(), Tensor(), ActivationType::LINEAR);

		output_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(output_fp32, output_fp16), 1.0e-3f);
	}
	TEST(TestWinograd3x3_4x4, cpuWeightTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		Tensor weight( { 31, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);

		Tensor matrices_fp32( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradWeightTransform(Context(), weight, matrices_fp32, false, false);

		weight.convertTo(Context(), DataType::FLOAT16);
		winogradWeightTransform(Context(), weight, matrices_fp16, false, false);

		matrices_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}

	TEST(TestWinograd5x5_2x2, cpuInputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		const Shape weights_shape( { 35, 5, 5, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0f);

		Tensor matrices_fp32( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradInputTransform(Context(), weights_shape, input, matrices_fp32);

		input.convertTo(Context(), DataType::FLOAT16);
		winogradInputTransform(Context(), weights_shape, input, matrices_fp16);

		matrices_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}
	TEST(TestWinograd5x5_2x2, cpuOutputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		const Shape weights_shape( { 35, 5, 5, 1 });

		Tensor matrices( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(matrices, 0.0f);

		Tensor output_fp32( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor output_fp16(output_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradOutputTransform(Context(), weights_shape, matrices, output_fp32, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.convertTo(Context(), DataType::FLOAT16);
		winogradOutputTransform(Context(), weights_shape, matrices, output_fp16, Tensor(), Tensor(), ActivationType::LINEAR);

		output_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(output_fp32, output_fp16), 1.0e-3f);
	}
	TEST(TestWinograd5x5_2x2, cpuWeightTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();

		Tensor weight( { 31, 3, 3, 35 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(weight, 0.0f);

		Tensor matrices_fp32( { 36, 31, 35 }, DataType::FLOAT32, Device::cpu());
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cpu());
		winogradWeightTransform(Context(), weight, matrices_fp32, false, false);

		weight.convertTo(Context(), DataType::FLOAT16);
		winogradWeightTransform(Context(), weight, matrices_fp16, false, false);

		matrices_fp16.convertTo(Context(), DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}

	TEST(TestWinograd3x3_4x4, cudaInputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		const Shape weights_shape( { 35, 3, 3, 1 });

		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(input, 0.0f);

		Tensor matrices_fp32( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradInputTransform(context, weights_shape, input, matrices_fp32);

		input.convertTo(context, DataType::FLOAT16);
		winogradInputTransform(context, weights_shape, input, matrices_fp16);

		matrices_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}
	TEST(TestWinograd3x3_4x4, cudaOutputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		const Shape weights_shape( { 35, 3, 3, 1 });

		Tensor matrices( { 36, 5 * 2 * 3, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(matrices, 0.0f);

		Tensor output_fp32( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor output_fp16(output_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradOutputTransform(context, weights_shape, matrices, output_fp32, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.convertTo(context, DataType::FLOAT16);
		winogradOutputTransform(context, weights_shape, matrices, output_fp16, Tensor(), Tensor(), ActivationType::LINEAR);

		output_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(output_fp32, output_fp16), 1.0e-3f);
	}
	TEST(TestWinograd3x3_4x4, cudaWeightTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		Tensor weight( { 31, 3, 3, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(weight, 0.0f);

		Tensor matrices_fp32( { 36, 31, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradWeightTransform(context, weight, matrices_fp32, false, false);

		weight.convertTo(context, DataType::FLOAT16);
		winogradWeightTransform(context, weight, matrices_fp16, false, false);

		matrices_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}

	TEST(TestWinograd5x5_2x2, cudaInputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		const Shape weights_shape( { 35, 5, 5, 1 });
		Tensor input( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(input, 0.0f);

		Tensor matrices_fp32( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradInputTransform(context, weights_shape, input, matrices_fp32);

		input.convertTo(context, DataType::FLOAT16);
		winogradInputTransform(context, weights_shape, input, matrices_fp16);

		matrices_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}
	TEST(TestWinograd5x5_2x2, cudaOutputTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		const Shape weights_shape( { 35, 5, 5, 1 });

		Tensor matrices( { 36, 5 * 4 * 6, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(matrices, 0.0f);

		Tensor output_fp32( { 5, 7, 11, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor output_fp16(output_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradOutputTransform(context, weights_shape, matrices, output_fp32, Tensor(), Tensor(), ActivationType::LINEAR);

		matrices.convertTo(context, DataType::FLOAT16);
		winogradOutputTransform(context, weights_shape, matrices, output_fp16, Tensor(), Tensor(), ActivationType::LINEAR);

		output_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(output_fp32, output_fp16), 1.0e-3f);
	}
	TEST(TestWinograd5x5_2x2, cudaWeightTransform_fp16)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context(Device::cuda(0));
		Tensor weight( { 31, 5, 5, 35 }, DataType::FLOAT32, Device::cuda(0));
		testing::initForTest(weight, 0.0f);

		Tensor matrices_fp32( { 36, 31, 35 }, DataType::FLOAT32, Device::cuda(0));
		Tensor matrices_fp16(matrices_fp32.shape(), DataType::FLOAT16, Device::cuda(0));
		winogradWeightTransform(context, weight, matrices_fp32, false, false);

		weight.convertTo(context, DataType::FLOAT16);
		winogradWeightTransform(context, weight, matrices_fp16, false, false);

		matrices_fp16.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(matrices_fp32, matrices_fp16), 1.0e-3f);
	}

} /* namespace ml */
