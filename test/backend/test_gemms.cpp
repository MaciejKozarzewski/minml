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
#include <minml/utils/testing_util.hpp>

#include <gtest/gtest.h>

namespace
{
	using namespace ml;
	void baseline_gemm_AB(Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
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
				C.set(tmp, { m, n });
			}
	}
	void baseline_gemm_ABT(Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
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
				C.set(tmp, { m, n });
			}
	}
	void baseline_gemm_ATB(Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
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
				C.set(tmp, { m, n });
			}
	}
	void baseline_gemm_ATBT(Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
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
				C.set(tmp, { m, n });
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

			GemmTester(int M, int N, int K, char opA, char opB, DataType dtype, DataType compute_type) :
					op_A(opA),
					op_B(opB)
			{
				assert(opA == 'n' || opA == 't');
				assert(opB == 'n' || opB == 't');
				Shape sh_A = (opA == 'n') ? Shape( { M, K }) : Shape( { K, M });
				Shape sh_B = (opB == 'n') ? Shape( { K, N }) : Shape( { N, K });
				A = Tensor(sh_A, dtype, Device::cpu());
				B = Tensor(sh_B, dtype, Device::cpu());

				ml::testing::initForTest(A, 0.0);
				ml::testing::initForTest(B, 1.57);

				C_baseline = Tensor( { M, N }, compute_type, Device::cpu());
				C_tested = Tensor( { M, N }, compute_type, Device::cpu());
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
						baseline_gemm_AB(C_baseline, A, B, alpha, beta);
					else
						baseline_gemm_ABT(C_baseline, A, B, alpha, beta);
				}
				else
				{
					if (op_B == 'n')
						baseline_gemm_ATB(C_baseline, A, B, alpha, beta);
					else
						baseline_gemm_ATBT(C_baseline, A, B, alpha, beta);
				}
			}
			double getDifference() const noexcept
			{
				return ml::testing::diffForTest(C_baseline, C_tested);
			}
			void moveTo(Device device)
			{
				A.moveTo(device);
				B.moveTo(device);
				C_tested.moveTo(device);
			}
	};
}

namespace ml
{

	TEST(TestGemmOnCPU, float32_AB)
	{
		GemmTester data(23, 45, 67, 'n', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float32_ABT)
	{
		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float32_ATB)
	{
		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float32_ATBT)
	{
		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}

	TEST(TestGemmOnCPU, float16_AB)
	{
		if (not Device::cpu().supportsType(DataType::FLOAT16))
			GTEST_SKIP_("CPU does not support fp16");

		GemmTester data(23, 45, 67, 'n', 'n', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float16_ABT)
	{
		if (not Device::cpu().supportsType(DataType::FLOAT16))
			GTEST_SKIP_("CPU does not support fp16");

		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float16_ATB)
	{
		if (not Device::cpu().supportsType(DataType::FLOAT16))
			GTEST_SKIP_("CPU does not support fp16");

		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCPU, float16_ATBT)
	{
		if (not Device::cpu().supportsType(DataType::FLOAT16))
			GTEST_SKIP_("CPU does not support fp16");

		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		gemm(Context(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}

	TEST(TestGemmOnCUDA, float32_AB)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 'n', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCUDA, float32_ABT)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 'n', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCUDA, float32_ATB)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 't', 'n', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
	TEST(TestGemmOnCUDA, float32_ATBT)
	{
		if (Device::numberOfCudaDevices() == 0)
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 't', 't', DataType::FLOAT32);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-4);
	}

	TEST(TestGemmOnCUDA, float16_AB)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 'n', 'n', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-3);
	}
	TEST(TestGemmOnCUDA, float16_ABT)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 'n', 't', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 2.0e-2);
	}
	TEST(TestGemmOnCUDA, float16_ATB)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 't', 'n', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 2.0e-3);
	}
	TEST(TestGemmOnCUDA, float16_ATBT)
	{
		if (Device::numberOfCudaDevices() == 0 or not Device::cuda(0).supportsType(DataType::FLOAT16))
			GTEST_SKIP();
		GemmTester data(23, 29, 37, 't', 't', DataType::FLOAT16);
		data.gemm_baseline(1.1, 0.1);

		data.moveTo(Device::cuda(0));
		gemm(Context(Device::cuda(0)), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
		EXPECT_LT(data.getDifference(), 1.0e-3);
	}

} /* namespace ml */

