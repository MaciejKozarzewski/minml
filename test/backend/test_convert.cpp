/*
 * test_convert.cpp
 *
 *  Created on: Feb 16, 2023
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

#include <iostream>

namespace
{
	using namespace ml;

	template<typename T>
	void transpose(void *dst, const void *src, int dim0, int dim1, int dim2)
	{
		for (int i = 0; i < dim0; i++)
			for (int j = 0; j < dim1; j++)
				for (int k = 0; k < dim2; k++)
				{
					const int src_idx = (i * dim1 + j) * dim2 + k;
					const int dst_idx = (i * dim2 + k) * dim1 + j;
					reinterpret_cast<T*>(dst)[dst_idx] = reinterpret_cast<const T*>(src)[src_idx];
				}
	}

	void unpack_input_fp32(float *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = (mask & 1u) ? 1.0f : 0.0f;
		}
	}
	void unpack_input_fp16(uint16_t *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = (mask & 1u) ? 0x3c00 : 0x000;
		}
	}

//	ml::Shape transpose(const ml::Shape &shape)
//	{
//		return ml::Shape( { shape[0], shape[2], shape[1] });
//	}

	int get_patch_size(int smaller, int larger) noexcept
	{
		assert(smaller <= larger);
		for (int i = 1;; i++)
		{
			const int tmp = (larger + i - 1) / i;
			if (tmp == smaller)
				return i;
			if (tmp < smaller)
				break;
		}
		return 0;
	}
	template<typename T>
	void baseline_space_to_depth(Tensor &output, const Tensor &input)
	{
		assert(output.firstDim() == input.firstDim());
		const int batch_size = input.firstDim();
		const int height = input.dim(1);
		const int width = input.dim(2);
		const int patch_size_h = get_patch_size(output.dim(1), input.dim(1));
		const int patch_size_w = get_patch_size(output.dim(2), input.dim(2));
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = input.lastDim();
		const int channels_out = output.lastDim();
		assert(channels_in * patch_size_h * patch_size_w == channels_out);

		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += patch_size_h)
				for (int w = 0; w < width; w += patch_size_w)
				{
					int c_out = 0;
					for (int x = 0; x < patch_size_h; x++)
						for (int y = 0; y < patch_size_w; y++)
							for (int c = 0; c < channels_in; c++, c_out++)
							{
								if ((h + x) < height and (w + y) < width)
									output.at( { b, h / patch_size_h, w / patch_size_w, c_out }) = (T) input.at( { b, h + x, w + y, c });
								else
									output.at( { b, h / patch_size_h, w / patch_size_w, c_out }) = (T) 0;
							}
				}
	}
	template<typename T>
	void baseline_depth_to_space(Tensor &output, const Tensor &input)
	{
		assert(output.firstDim() == input.firstDim());
		const int batch_size = input.firstDim();
		const int height = output.dim(1);
		const int width = output.dim(2);
		const int patch_size_h = get_patch_size(input.dim(1), output.dim(1));
		const int patch_size_w = get_patch_size(input.dim(2), output.dim(2));
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = input.lastDim();
		const int channels_out = output.lastDim();
		assert(channels_out * patch_size_h * patch_size_w == channels_in);

		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += patch_size_h)
				for (int w = 0; w < width; w += patch_size_w)
				{
					int c_in = 0;
					for (int x = 0; x < patch_size_h; x++)
						for (int y = 0; y < patch_size_w; y++)
							for (int c = 0; c < channels_out; c++, c_in++)
							{
								if ((h + x) < height and (w + y) < width)
									output.at( { b, h + x, w + y, c }) = (T) input.at( { b, h / patch_size_h, w / patch_size_w, c_in });
							}
				}
	}

	class BaselineSpaceToDepth: public Layer
	{
			int m_patch_size_h, m_patch_size_w;
		public:
			BaselineSpaceToDepth(int patch_size) :
					Layer(),
					m_patch_size_h(patch_size),
					m_patch_size_w(patch_size)
			{
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int height = (getInputShape().dim(1) + m_patch_size_h - 1) / m_patch_size_h;
				const int width = (getInputShape().dim(2) + m_patch_size_w - 1) / m_patch_size_w;
				const int channels = getInputShape().dim(3) * m_patch_size_h * m_patch_size_w;
				return Shape( { batch_size, height, width, channels });
			}
			std::string name() const
			{
				return "BaselineSpaceToDepth";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["patch_size_h"] = m_patch_size_h;
				result["patch_size_w"] = m_patch_size_w;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineSpaceToDepth> result = std::make_unique<BaselineSpaceToDepth>(config["patch_size_h"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				switch (dtype())
				{
					default:
						break;
					case DataType::FLOAT32:
						baseline_space_to_depth<float>(output, input[0]);
						break;
					case DataType::FLOAT64:
						baseline_space_to_depth<double>(output, input[0]);
						break;
				}
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
			{
				switch (dtype())
				{
					default:
						break;
					case DataType::FLOAT32:
						baseline_depth_to_space<float>(gradient_prev[0], gradient_next);
						break;
					case DataType::FLOAT64:
						baseline_depth_to_space<double>(gradient_prev[0], gradient_next);
						break;
				}
			}
	};
	class BaselineDepthToSpace: public Layer
	{
			int m_patch_size_h, m_patch_size_w;
			Shape m_output_shape;
		public:
			BaselineDepthToSpace(int patch_size, const Shape &output_shape) :
					Layer(),
					m_patch_size_h(patch_size),
					m_patch_size_w(patch_size),
					m_output_shape(output_shape)
			{
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int height = m_output_shape.dim(0);
				const int width = m_output_shape.dim(1);
				const int channels = getInputShape().lastDim() / (m_patch_size_h * m_patch_size_w);
				return Shape( { batch_size, height, width, channels });
			}
			std::string name() const
			{
				return "BaselineDepthToSpace";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["patch_size_h"] = m_patch_size_h;
				result["patch_size_w"] = m_patch_size_w;
				result["output_shape"] = m_output_shape.serialize();
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineDepthToSpace> result = std::make_unique<BaselineDepthToSpace>(config["patch_size_h"].getInt(),
						Shape(config["output_shape"]));
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				switch (dtype())
				{
					default:
						break;
					case DataType::FLOAT32:
						baseline_depth_to_space<float>(output, input[0]);
						break;
					case DataType::FLOAT64:
						baseline_depth_to_space<double>(output, input[0]);
						break;
				}
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
			{
				switch (dtype())
				{
					default:
						break;
					case DataType::FLOAT32:
						baseline_space_to_depth<float>(gradient_prev[0], gradient_next);
						break;
					case DataType::FLOAT64:
						baseline_space_to_depth<double>(gradient_prev[0], gradient_next);
						break;
				}
			}
	};
}

namespace ml
{
	TEST(TestSpaceToDepth, baseline)
	{
//		testing::GradientCheck gradcheck { BaselineSpaceToDepth(4) };
//		gradcheck.setInputShape(Shape( { 10, 15, 15, 32 }));

//		testing::GradientCheck gradcheck { BaselineDepthToSpace(2, { 15, 15 }) };
//		gradcheck.setInputShape(Shape( { 10, 8, 8, 32 }));
//
//		gradcheck.check(1000, 1.0e-3, "all");
	}

	TEST(TestSpaceToDepth, forward)
	{
		const int batch_size = 12;
		const int height = 15;
		const int width = 17;
		const int channels = 8;
		const int patch_size = 4;

		const Shape input_shape( { batch_size, height, width, channels });
		const Shape output_shape(
				{ batch_size, (height + patch_size - 1) / patch_size, (width + patch_size - 1) / patch_size, channels * patch_size * patch_size });

		Tensor input(input_shape, "float32", Device::cpu());
		Tensor output(output_shape, "float32", Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0);

		baseline_space_to_depth<float>(correct_output, input);

		spaceToDepth(Context(), input, output);

		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			spaceToDepth(context, input, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}

	TEST(TestDepthToSpace, forward)
	{
		const int batch_size = 12;
		const int height = 15;
		const int width = 17;
		const int channels = 8;
		const int patch_size = 3;

		const Shape input_shape(
				{ batch_size, (height + patch_size - 1) / patch_size, (width + patch_size - 1) / patch_size, channels * patch_size * patch_size });
		const Shape output_shape( { batch_size, height, width, channels });

		Tensor input(input_shape, "float32", Device::cpu());
		Tensor output(output_shape, "float32", Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0);

		baseline_depth_to_space<float>(correct_output, input);

		depthToSpace(Context(), input, output);

		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();

			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			depthToSpace(context, input, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}

	TEST(TestUnpackInput, cpu_fp32)
	{
		Context context;
		Tensor input(Shape( { 11, 12, 13, 1 }), DataType::INT32, Device::cpu());
		for (int i = 0; i < input.volume(); i++)
			reinterpret_cast<int32_t*>(input.data())[i] = i;

//		Tensor output(Shape( { 11, 12, 13, 32 }), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
	}

//	TEST(TestTranspose021, cpu_fp32)
//	{
//		Context context;
//		Tensor input(Shape( { 14, 76, 45 }), DataType::FLOAT32, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor output(transpose(input.shape()), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//	TEST(TestTranspose021, cpu_fp16)
//	{
//		Context context;
//		Tensor input(Shape( { 14, 76, 45 }), DataType::FLOAT16, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor output(transpose(input.shape()), input.dtype(), input.device());
//		Tensor correct_output(output.shape(), input.dtype(), input.device());
//
//		transpose<uint16_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		transpose_021(context, input, output);
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//
//	TEST(TestTranspose021, cuda_fp32)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Context context(Device::cuda(0));
//		Tensor input(Shape( { 14, 176, 145 }), DataType::FLOAT32, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor correct_output(transpose(input.shape()), input.dtype(), Device::cpu());
//
//		transpose<uint32_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		input.moveTo(Device::cuda(0));
//		Tensor output(transpose(input.shape()), input.dtype(), Device::cuda(0));
//		transpose_021(context, input, output);
//		context.synchronize();
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}
//	TEST(TestTranspose021, cuda_fp16)
//	{
//		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
//			GTEST_SKIP();
//		Context context(Device::cuda(0));
//		Tensor input(Shape( { 14, 176, 145 }), DataType::FLOAT16, Device::cpu());
//		testing::initForTest(input, 0.0);
//
//		Tensor correct_output(transpose(input.shape()), input.dtype(), Device::cpu());
//
//		transpose<uint16_t>(correct_output.data(), input.data(), input.dim(0), input.dim(1), input.dim(2));
//
//		input.moveTo(Device::cuda(0));
//		Tensor output(transpose(input.shape()), input.dtype(), Device::cuda(0));
//		transpose_021(context, input, output);
//		context.synchronize();
//		EXPECT_LE(testing::diffForTest(output, correct_output), 1.0e-6f);
//	}

}
