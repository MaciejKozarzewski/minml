/*
 * test_tensor_op.cpp
 *
 *  Created on: Sep 20, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/json.hpp>

#include <gtest/gtest.h>
#include <cmath>

namespace
{
	using namespace ml;

	void baseline_add_bias_act(Tensor &dst, const Tensor &src, ActivationType act)
	{
		const int first_dim = dst.shape().volumeWithoutLastDim();
		const int last_dim = dst.shape().lastDim();
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
				dst.at( { i, j }) = (float) dst.at( { i, j }) + (float) src.at( { j });
		activationForward(Context(), 1.0f, dst, 0.0f, dst, act);
	}
	void baseline_sum_over_first_dim(float beta, Tensor &dst, float alpha, const Tensor &src)
	{
		const int last_dim = dst.volume();
		const int first_dim = src.volume() / last_dim;
		for (int j = 0; j < last_dim; j++)
		{
			float acc = 0.0f;
			for (int i = 0; i < first_dim; i++)
				acc += src.get( { i, j });
			acc *= alpha;
			if (beta != 0.0f)
				acc += dst.get( { j }) * beta;
			dst.at( { j }) = acc;
		}
	}
	void baseline_add_tensors(Tensor &dst, const Tensor &src1, const Tensor &src2)
	{
		for (int i = 0; i < dst.volume(); i++)
			reinterpret_cast<float*>(dst.data())[i] = reinterpret_cast<const float*>(src1.data())[i] + reinterpret_cast<const float*>(src2.data())[i];
	}

	Tensor window_partition(const Tensor &src, const Shape &window_size, const Shape &offset)
	{
		assert(src.rank() == 4);

		const int batch_size = src.dim(0);
		const int height = src.dim(1);
		const int width = src.dim(2);
		const int channels = src.dim(3);
		const int num_windows_h = (height + window_size[0] - 1) / window_size[0];
		const int num_windows_w = (width + window_size[1] - 1) / window_size[1];

		Tensor result( { batch_size, num_windows_h, num_windows_w, window_size[0], window_size[1], channels }, src.dtype(), src.device());

		const uint8_t *src_ptr = reinterpret_cast<const uint8_t*>(src.data());
		uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(result.data());
		const size_t block_size = channels * sizeOf(src.dtype());

		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
				{
					const int x = (h + offset[0] + height) % height;
					const int y = (w + offset[1] + width) % width;

					const int window_idx_h = x / window_size[0];
					const int window_idx_w = y / window_size[1];

					const int idx_h = x % window_size[0];
					const int idx_w = y % window_size[1];

					std::memcpy(dst_ptr + sizeOf(src.dtype()) * result.getIndexOf( { b, window_idx_h, window_idx_w, idx_h, idx_w, 0 }),
							src_ptr + sizeOf(src.dtype()) * src.getIndexOf( { b, h, w, 0 }), block_size);
				}
		result.reshape( { batch_size * num_windows_h * num_windows_w, window_size[0], window_size[1], channels });
		return result;
	}
	Tensor window_merging(const Tensor &src, const Shape &dst_shape, const Shape &offset)
	{
		assert(src.rank() == 4);
		assert(dst_shape.rank() == 4);

		const int batch_size = dst_shape.dim(0);
		const int height = dst_shape.dim(1);
		const int width = dst_shape.dim(2);
		const int channels = dst_shape.dim(3);

		const int window_height = src.dim(1);
		const int window_width = src.dim(2);

		const int num_windows_h = (height + window_height - 1) / window_height;
		const int num_windows_w = (width + window_width - 1) / window_width;

		Tensor result(dst_shape, src.dtype(), src.device());

		const uint8_t *src_ptr = reinterpret_cast<const uint8_t*>(src.data());
		uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(result.data());
		const size_t block_size = channels * sizeOf(src.dtype());

		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
				{
					const int x = (h + offset[0] + height) % height;
					const int y = (w + offset[1] + width) % width;

					const int window_idx_h = x / window_height;
					const int window_idx_w = y / window_width;

					const int idx_h = x % window_height;
					const int idx_w = y % window_width;

					const int idx_b = b * num_windows_h * num_windows_w + window_idx_h * num_windows_w + window_idx_w * 1;

					std::memcpy(dst_ptr + sizeOf(src.dtype()) * result.getIndexOf( { b, h, w, 0 }), src_ptr + sizeOf(src.dtype()) * src.getIndexOf( {
							idx_b, idx_h, idx_w, 0 }), block_size);
				}

		return result;
	}

	class BaselineWindowPartition: public Layer
	{
			int m_window_size = 0;
			int m_window_shift = 0;
		public:
			BaselineWindowPartition(int window_size, int window_shift) :
					Layer(),
					m_window_size(window_size),
					m_window_shift(window_shift)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int num_windows_h = (getInputShape().dim(1) + m_window_size - 1) / m_window_size;
				const int num_windows_w = (getInputShape().dim(2) + m_window_size - 1) / m_window_size;
				const int channels = getInputShape().dim(3);
				return Shape( { batch_size * num_windows_h * num_windows_w, m_window_size, m_window_size, channels });
			}
			std::string name() const
			{
				return "BaselineWindowPartition";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["window_size"] = m_window_size;
				result["window_shift"] = m_window_shift;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineWindowPartition> result = std::make_unique<BaselineWindowPartition>(config["window_size"].getInt(),
						config["window_shift"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				output = window_partition(input[0], { m_window_size, m_window_size }, { m_window_shift, m_window_shift });
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				gradient_prev[0] = window_merging(gradient_next, getInputShape(), { m_window_shift, m_window_shift });
			}
	};
	class BaselineWindowMerging: public Layer
	{
			Shape m_dst_shape;
			int m_window_shift = 0;
		public:
			BaselineWindowMerging(const Shape &dst_shape, int window_shift) :
					Layer(),
					m_dst_shape(dst_shape),
					m_window_shift(window_shift)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return m_dst_shape;
			}
			std::string name() const
			{
				return "BaselineWindowMerging";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["dst_shape"] = m_dst_shape.serialize();
				result["window_shift"] = m_window_shift;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineWindowMerging> result = std::make_unique<BaselineWindowMerging>(Shape(config["dst_shape"]),
						config["window_shift"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				output = window_merging(input[0], getOutputShape(), { m_window_shift, m_window_shift });
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				const Shape window_size( { getInputShape().dim(1), getInputShape().dim(2) });
				gradient_prev[0] = window_partition(gradient_next, window_size, { m_window_shift, m_window_shift });
			}
	};
}

namespace ml
{
//	TEST(TestTensorOp, baseline)
//	{
////		testing::GradientCheck gradcheck { BaselineWindowPartition(5, 2) };
////		gradcheck.setInputShape(Shape( { 10, 15, 15, 32 }));
//
//		testing::GradientCheck gradcheck { BaselineWindowMerging(Shape( { 10, 15, 15, 32 }), 2) };
//		gradcheck.setInputShape(Shape( { 90, 5, 5, 32 }));
//
//		gradcheck.check(1000, 1.0e-2, "input");
//
//		exit(0);
//	}

	TEST(TestTensorOp, setall_cpu)
	{
		Tensor t( { 123 }, "float32", Device::cpu());
		t.setall(1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}
	TEST(TestTensorOp, setall_cuda)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP_("No CUDA devices");
		Tensor t( { 123 }, "float32", Device::cuda(0));
		t.setall(1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}
	TEST(TestTensorOp, setall_opencl)
	{
		if (Device::numberOfOpenCLDevices() == 0)
			GTEST_SKIP_("No OpenCL devices");
		Tensor t( { 123 }, "float32", Device::opencl(0));
		t.setall(1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}

	TEST(TestTensorOp, addBiasAct_fp32)
	{
		Context context;
		const int channels = 48;

		Tensor correct_output( { 123, channels }, "float32", Device::cpu());
		Tensor output = zeros_like(correct_output);
		Tensor bias( { channels }, "float32", Device::cpu());
		testing::initForTest(bias, 0.0f);
		testing::initForTest(output, 1.0f);
		testing::initForTest(correct_output, 1.0f);

		baseline_add_bias_act(correct_output, bias, ActivationType::SIGMOID);

		addBiasAct(context, 1.0f, output, bias, 0.0f, output, ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.0f);
			bias.moveTo(device);
			output.moveTo(device);
			addBiasAct(context, 1.0f, output, bias, 0.0f, output, ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestTensorOp, addBiasAct_fp16)
	{
		if (not Device::cpu().supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		Context context;
		const int channels = 48;
		Tensor correct_output( { 123, channels }, "float16", Device::cpu());
		Tensor output = zeros_like(correct_output);
		Tensor bias( { channels }, "float16", Device::cpu());
		testing::initForTest(bias, 0.0f);
		testing::initForTest(output, 1.0f);
		testing::initForTest(correct_output, 1.0f);

		baseline_add_bias_act(correct_output, bias, ActivationType::SIGMOID);

		addBiasAct(context, 1.0f, output, bias, 0.0f, output, ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.0f);
			bias.moveTo(device);
			output.moveTo(device);
			addBiasAct(context, 1.0f, output, bias, 0.0f, output, ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}
	}

	TEST(TestTensorOp, sumOverFirstDim_fp32)
	{
		const float alpha = 1.1f;
		const float beta = 0.1f;
		Tensor src( { 256 * 15 * 15, 36 });
		Tensor dst( { 36 });
		testing::initForTest(src, 0.0f);
		testing::initForTest(dst, 1.0f);

		Tensor correct_dst = dst;
		baseline_sum_over_first_dim(beta, correct_dst, alpha, src);

//		sumOverFirstDim(Context(), alpha, src, beta, dst);
//		EXPECT_LE(testing::diffForTest(correct_dst, dst), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(dst, 1.0f);
			src.moveTo(device);
			dst.moveTo(device);
			sumOverFirstDim(context, alpha, src, beta, dst);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_dst, dst), 1.0e-4);
		}
	}
	TEST(TestTensorOp, sumOverFirstDim_fp16)
	{
		const float alpha = 1.1f;
		const float beta = 0.1f;
		Tensor src( { 256 * 15 * 15, 35 }, "float16", Device::cpu());
		Tensor dst( { 35 }, "float16", Device::cpu());
		testing::initForTest(src, 0.0f);
		testing::initForTest(dst, 1.0f);

		Tensor correct_dst = dst;
		baseline_sum_over_first_dim(beta, correct_dst, alpha, src);

//		sumOverFirstDim(Context(), alpha, src, beta, dst);
//		EXPECT_LE(testing::diffForTest(correct_dst, dst), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(dst, 1.0f);
			src.moveTo(device);
			dst.moveTo(device);
			sumOverFirstDim(context, alpha, src, beta, dst);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_dst, dst), 1.0e-3);
		}
	}

	TEST(TestTensorOp, addTensors)
	{
		Context context;
		Tensor correct( { 123, 14 }, DataType::FLOAT32, Device::cpu());
		Tensor dst( { 123, 14 }, DataType::FLOAT32, Device::cpu());

		Tensor src1( { 123, 14 }, DataType::FLOAT32, Device::cpu());
		Tensor src2( { 123, 14 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(src1, 0.0f);
		testing::initForTest(src2, 1.0f);

		baseline_add_tensors(correct, src1, src2);
		addTensors(context, dst, src1, src2);
		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			src1.moveTo(device);
			src2.moveTo(device);
			dst.moveTo(device);
			dst.zeroall();
			addTensors(context, dst, src1, src2);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
		}
	}

	TEST(TestTensorOp, window_partition)
	{
		const Shape window_size( { 3, 4 });
		const Shape window_offset( { 1, 2 });

		Context context;
		Tensor input( { 12, 13, 14, 15 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0);
		Tensor correct_output = window_partition(input, window_size, window_offset);

		Tensor output = zeros_like(correct_output);

//		EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			windowPartitioning(context, input, output, window_offset);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);
		}
	}
	TEST(TestTensorOp, window_merging)
	{
		const Shape window_size( { 3, 4 });
		const Shape window_offset( { -1, -2 });

		Context context;
		Tensor input( { 12 * 5 * 4, 3, 4, 15 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(input, 0.0);
		Tensor correct_output = window_merging(input, Shape( { 12, 13, 14, 15 }), window_offset);

		Tensor output = zeros_like(correct_output);

//		EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			windowMerging(context, input, output, window_offset);
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);
		}
	}

	TEST(TestTensorOp, transpose2D)
	{
		Tensor input( { 10, 23 });
		Tensor output( { 23, 10 });
		testing::initForTest(input, 0.0f);

		Tensor correct_output = zeros_like(output);
		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				correct_output.at( { j, i }) = (float) input.at( { i, j });

//		transpose(Context(), output, input, { 1, 0 });
//		EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);

		if (Device::numberOfCudaDevices() > 0)
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			transpose(context, output, input, { 1, 0 });
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);
		}
	}
	TEST(TestTensorOp, transpose3D)
	{
		Tensor input( { 10, 15, 23 });
		Tensor output( { 23, 10, 15 });
		testing::initForTest(input, 0.0f);

		Tensor correct_output = zeros_like(output);
		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					correct_output.at( { k, i, j }) = (float) input.at( { i, j, k });

//		transpose(Context(), output, input, { 2, 0, 1 });
//		EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);

		if (Device::numberOfCudaDevices() > 0)
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			transpose(context, output, input, { 2, 0, 1 });
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);
		}
	}
	TEST(TestTensorOp, transpose4D)
	{
		Tensor input( { 10, 15, 20, 23 });
		Tensor output( { 20, 10, 23, 15 });
		testing::initForTest(input, 0.0f);

		Tensor correct_output = zeros_like(output);
		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
						correct_output.at( { k, i, l, j }) = (float) input.at( { i, j, k, l });

//		transpose(Context(), output, input, { 2, 0, 3, 1 });
//		EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);

		if (Device::numberOfCudaDevices() > 0)
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			transpose(context, output, input, { 2, 0, 3, 1 });
			context.synchronize();
			EXPECT_EQ(testing::diffForTest(correct_output, output), 0.0);
		}
	}

}
/* namespace ml */

