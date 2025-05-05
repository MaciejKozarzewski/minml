/*
 * test_gemms.cpp
 *
 *  Created on: Sep 12, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/testing_util.hpp>

#include <gtest/gtest.h>

namespace
{
	using namespace ml;
	void add_bias_relu(Tensor &D, const Tensor &bias, bool use_relu)
	{
		for (int m = 0; m < D.dim(0); m++)
			for (int n = 0; n < D.dim(1); n++)
			{
				float tmp = D.get( { m, n }) + bias.get( { n });
				if (use_relu)
					tmp = std::max(0.0f, tmp);
				D.set(tmp, { m, n });
			}
	}
	void baseline_gemm_NN(Tensor &D, const Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		assert(A.device().isCPU());
		for (int m = 0; m < A.dim(0); m++)
			for (int n = 0; n < B.dim(1); n++)
			{
				float tmp = 0.0f;
				for (int k = 0; k < A.dim(1); k++)
					tmp += A.get( { m, k }) * B.get( { k, n });
				tmp = alpha * tmp;
				if (beta != 0.0f)
					tmp += beta * C.get( { m, n });
				D.set(tmp, { m, n });
			}
	}
	void baseline_gemm_NT(Tensor &D, const Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		assert(A.device().isCPU());
		for (int m = 0; m < A.dim(0); m++)
			for (int n = 0; n < B.dim(0); n++)
			{
				float tmp = 0.0f;
				for (int k = 0; k < A.dim(1); k++)
					tmp += A.get( { m, k }) * B.get( { n, k });
				tmp = alpha * tmp;
				if (beta != 0.0f)
					tmp += beta * C.get( { m, n });
				D.set(tmp, { m, n });
			}
	}
	void baseline_gemm_TN(Tensor &D, const Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		assert(A.device().isCPU());
		for (int m = 0; m < A.dim(1); m++)
			for (int n = 0; n < B.dim(1); n++)
			{
				float tmp = 0.0f;
				for (int k = 0; k < A.dim(0); k++)
					tmp += A.get( { k, m }) * B.get( { k, n });
				tmp = alpha * tmp;
				if (beta != 0.0f)
					tmp += beta * C.get( { m, n });
				D.set(tmp, { m, n });
			}
	}
	void baseline_gemm_TT(Tensor &D, const Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		assert(A.device().isCPU());
		for (int m = 0; m < A.dim(1); m++)
			for (int n = 0; n < B.dim(0); n++)
			{
				float tmp = 0.0f;
				for (int k = 0; k < A.dim(0); k++)
					tmp += A.get( { k, m }) * B.get( { n, k });
				tmp = alpha * tmp;
				if (beta != 0.0f)
					tmp += beta * C.get( { m, n });
				D.set(tmp, { m, n });
			}
	}

	class GemmTester
	{
		private:
			char op_A;
			char op_B;
		public:

			Tensor A;
			Tensor B;
			Tensor C_baseline;
			Tensor C_tested;
			Tensor D_tested;
			Tensor bias;

			GemmTester(int M, int N, int K, char opA, char opB, DataType dtype, DataType compute_type) :
					op_A(opA),
					op_B(opB)
			{
				assert(opA == 'n' || opA == 't');
				assert(opB == 'n' || opB == 't');
				const Shape sh_A = (opA == 'n') ? Shape( { M, K }) : Shape( { K, M });
				const Shape sh_B = (opB == 'n') ? Shape( { K, N }) : Shape( { N, K });
				A = Tensor(sh_A, dtype, Device::cpu());
				B = Tensor(sh_B, dtype, Device::cpu());

				ml::testing::initForTest(A, 0);
				ml::testing::initForTest(B, 1);

				C_baseline = Tensor( { M, N }, compute_type, Device::cpu());
				C_tested = zeros_like(C_baseline);
				D_tested = zeros_like(C_baseline);

				bias = Tensor( { N }, compute_type, Device::cpu());
				ml::testing::initForTest(C_baseline, 2);
				ml::testing::initForTest(C_tested, 2);
				ml::testing::initForTest(bias, 3);
			}
			GemmTester(int M, int N, int K, char opA, char opB, DataType dtype) :
					GemmTester(M, N, K, opA, opB, dtype, dtype)
			{
			}
			void gemm_baseline(float alpha, float beta) noexcept
			{
				if (op_A == 'n')
				{
					if (op_B == 'n')
						baseline_gemm_NN(C_baseline, C_baseline, A, B, alpha, beta);
					else
						baseline_gemm_NT(C_baseline, C_baseline, A, B, alpha, beta);
				}
				else
				{
					if (op_B == 'n')
						baseline_gemm_TN(C_baseline, C_baseline, A, B, alpha, beta);
					else
						baseline_gemm_TT(C_baseline, C_baseline, A, B, alpha, beta);
				}
			}
			void gemm_ex_baseline(float alpha, float beta) noexcept
			{
				gemm_baseline(alpha, beta);
				add_bias_relu(C_baseline, bias, true);
			}
			double getDifference() const noexcept
			{
				return ml::testing::diffForTest(C_baseline, C_tested);
			}
			double getDifferenceEx() const noexcept
			{
				return ml::testing::diffForTest(C_baseline, D_tested);
			}
			void moveTo(Device device)
			{
				A.moveTo(device);
				B.moveTo(device);
				ml::testing::initForTest(C_tested, 2);
				C_tested.moveTo(device);
				D_tested.moveTo(device);
				bias.moveTo(device);
			}
	};
}

namespace ml
{

	TEST(TestGemm, float32_NN)
	{
		GemmTester data(23, 45, 67, 'n', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1f);

		gemm(Context(), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1f);
		EXPECT_LT(data.getDifference(), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float32_NT)
	{
		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float32_TN)
	{
		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float32_TT)
	{
		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}

	TEST(TestGemm, float16_NN)
	{
		GemmTester data(23, 45, 67, 'n', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			gemm(Context(), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float16_NT)
	{
		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			gemm(Context(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float16_TN)
	{
		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			gemm(Context(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}
	TEST(TestGemm, float16_TT)
	{
		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			gemm(Context(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm(context, 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
			context.synchronize();
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
	}

	TEST(TestGemmEx, float32_NT)
	{
		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
		data.gemm_ex_baseline(1.1, 0.1);

		gemm_ex(Context(), data.D_tested, 1.1f, 'n', data.A, 't', data.B, 0.1f, data.C_tested, data.bias, ActivationType::RELU);
		EXPECT_LT(data.getDifferenceEx(), 1.0e-4);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm_ex(context, data.D_tested, 1.1f, 'n', data.A, 't', data.B, 0.1f, data.C_tested, data.bias, ActivationType::RELU);
			context.synchronize();
			EXPECT_LT(data.getDifferenceEx(), 1.0e-4);
		}
	}
	TEST(TestGemmEx, float16_NT)
	{
		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
		data.gemm_ex_baseline(1.1, 0.1);

		if (Device::cpu().supportsType(DataType::FLOAT16))
		{
			gemm_ex(Context(), data.D_tested, 1.1f, 'n', data.A, 't', data.B, 0.1f, data.C_tested, data.bias, ActivationType::RELU);
			EXPECT_LT(data.getDifferenceEx(), 1.0e-4);
		}

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			data.moveTo(context.device());
			gemm_ex(context, data.D_tested, 1.1f, 'n', data.A, 't', data.B, 0.1f, data.C_tested, data.bias, ActivationType::RELU);
			context.synchronize();
			EXPECT_LT(data.getDifferenceEx(), 1.0e-4);
		}
	}

} /* namespace ml */

