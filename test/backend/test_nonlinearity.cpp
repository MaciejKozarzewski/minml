/*
 * test_nonlinearity.cpp
 *
 *  Created on: Mar 9, 2021
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/ml_memory.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/layers/Layer.hpp>

namespace
{
	ml::Tensor to_tensor(const std::vector<float> &vec)
	{
		ml::Tensor result(ml::Shape( { (int) vec.size() }), "float32", ml::Device::cpu());
		ml::memcpy(result.device(), result.data(), 0, ml::Device::cpu(), vec.data(), 0, sizeof(float) * vec.size());
		return result;
	}
}

namespace ml
{

	TEST(TestSoftmax, ForwardOnCPU)
	{
		Tensor input = to_tensor( { { 0.1f, -0.9f, 2.0f, 0.0f, 0.3f, -1.0f, 0.7f, -0.1f } });
		input.reshape(Shape( { 2, 4 }));

		Tensor correct_output = to_tensor( {
				{ 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f, 0.29114823f, 0.07934714f, 0.43434212f, 0.19516249f } });
		correct_output.reshape(Shape( { 2, 4 }));

		activationForwardInPlace(Context(), input, ActivationType::SOFTMAX);
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-4f);
	}
	TEST(TestSoftmax, ForwardOnCUDA)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP_("No CUDA enabled devices");
		Tensor input = to_tensor( { { 0.1f, -0.9f, 2.0f, 0.0f, 0.3f, -1.0f, 0.7f, -0.1f } });
		input.reshape(Shape( { 2, 4 }));

		Tensor correct_output = to_tensor( {
				{ 0.11162444f, 0.04106433f, 0.74630924f, 0.10100197f, 0.29114823f, 0.07934714f, 0.43434212f, 0.19516249f } });
		correct_output.reshape(Shape( { 2, 4 }));

		activationForwardInPlace(Context(), input, ActivationType::SOFTMAX);
		EXPECT_LE(testing::diffForTest(input, correct_output), 1.0e-4f);
	}
} /* namespace ml */

