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

#include <gtest/gtest.h>
#include <cmath>

namespace
{
	using namespace ml;

	void baseline_add_bias_act(Tensor &dst, const Tensor &src, ActivationType act)
	{
		assert(same_device(dst, src));
		assert(dst.device().isCPU());
		const int first_dim = dst.shape().volumeWithoutLastDim();
		const int last_dim = dst.shape().lastDim();
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
				reinterpret_cast<float*>(dst.data())[i * last_dim + j] += reinterpret_cast<const float*>(src.data())[j];
		activationForward(Context(), dst, dst, act);
	}
	void baseline_sum_over_first_dim(Tensor &dst, const Tensor &src, float beta)
	{
		assert(same_device(dst, src));
		assert(dst.device().isCPU());

		const int first_dim = src.shape().volumeWithoutLastDim();
		const int last_dim = src.shape().lastDim();
		for (int j = 0; j < last_dim; j++)
		{
			float acc = 0.0f;
			for (int i = 0; i < first_dim; i++)
				acc += reinterpret_cast<const float*>(src.data())[i * last_dim + j];
			if (beta == 0)
				reinterpret_cast<float*>(dst.data())[j] = acc;
			else
				reinterpret_cast<float*>(dst.data())[j] = reinterpret_cast<float*>(dst.data())[j] * beta + acc;
		}
	}
	void baseline_add_tensors(Tensor &dst, const Tensor &src1, const Tensor &src2)
	{
		for (int i = 0; i < dst.volume(); i++)
			reinterpret_cast<float*>(dst.data())[i] = reinterpret_cast<const float*>(src1.data())[i] + reinterpret_cast<const float*>(src2.data())[i];
	}
}

namespace ml
{
	TEST(TestTensorOp, setall_cpu)
	{
		Tensor t( { 123 }, "float32", Device::cpu());
		t.setall(Context(), 1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}
	TEST(TestTensorOp, setall_cuda)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP_("No CUDA devices");
		Tensor t( { 123 }, "float32", Device::cuda(0));
		t.setall(Context(Device::cuda(0)), 1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}
	TEST(TestTensorOp, setall_opencl)
	{
		if (Device::numberOfOpenCLDevices() == 0)
			GTEST_SKIP_("No OpenCL devices");
		Tensor t( { 123 }, "float32", Device::opencl(0));
		t.setall(Context(Device::opencl(0)), 1.23f);

		for (int i = 0; i < t.volume(); i++)
			EXPECT_EQ(t.get( { i }), 1.23f);
	}

	TEST(TestTensorOp, addBiasAct_fp32)
	{
		Context context;
		Tensor correct_output( { 123, 34 }, DataType::FLOAT32, Device::cpu());
		Tensor output( { 123, 34 }, DataType::FLOAT32, Device::cpu());
		Tensor bias( { 34 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(bias, 0.0f);
		testing::initForTest(output, 1.57f);
		testing::initForTest(correct_output, 1.57f);

		baseline_add_bias_act(correct_output, bias, ActivationType::SIGMOID);

		addBiasAct(context, output, bias, ActivationType::SIGMOID);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.57f);
			bias.moveTo(device);
			output.moveTo(device);
			addBiasAct(context, output, bias, ActivationType::SIGMOID);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestTensorOp, addBiasAct_fp16)
	{
		Context context;
		Tensor correct_output( { 123, 34 }, DataType::FLOAT32, Device::cpu());
		Tensor output( { 123, 34 }, DataType::FLOAT16, Device::cpu());
		Tensor bias( { 34 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(bias, 0.0f);
		testing::initForTest(output, 1.57f);
		testing::initForTest(correct_output, 1.57f);

		baseline_add_bias_act(correct_output, bias, ActivationType::SIGMOID);

		bias.convertTo(context, DataType::FLOAT16);
		addBiasAct(context, output, bias, ActivationType::SIGMOID);
		output.convertTo(context, DataType::FLOAT32);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(output, 1.57f);
			bias.moveTo(device);
			output.moveTo(device);
			output.convertTo(context, DataType::FLOAT16);
			addBiasAct(context, output, bias, ActivationType::SIGMOID);
			output.convertTo(context, DataType::FLOAT32);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3);
		}
	}

	TEST(TestTensorOp, sumOverFirstDim)
	{
		float beta = 2.1f;
		Tensor src( { 256 * 15 * 15, 64 }, DataType::FLOAT32, Device::cpu());
		Tensor correct( { 64 }, DataType::FLOAT32, Device::cpu());
		Tensor dst( { 64 }, DataType::FLOAT32, Device::cpu());
		testing::initForTest(src, 0.0f);
		testing::initForTest(correct, 1.0f);
		testing::initForTest(dst, 1.0f);

		baseline_sum_over_first_dim(correct, src, beta);

		sumOverFirstDim(Context(), dst, src, beta);
		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			testing::initForTest(dst, 1.0f);
			src.moveTo(device);
			dst.moveTo(device);
			sumOverFirstDim(context, dst, src, beta);
			context.synchronize();
			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-4);
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

} /* namespace ml */

