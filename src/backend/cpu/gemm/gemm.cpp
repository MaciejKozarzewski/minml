/*
 * gemm.cpp
 *
 *  Created on: May 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "Matrix.hpp"
#include "gemm_runtime.hpp"

namespace
{
	using namespace ml;
	int get_batch_stride(const mlShape_t &shape)
	{
		switch (shape.rank)
		{
			default:
				throw std::logic_error("unsupported shape");
			case 2:
				return 0;
			case 3:
				return shape.dim[1] * shape.dim[2];
		}
	}
}

namespace ml
{
	void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		cpu_gemm_ex(context, dtype, shape_C, C, alpha, opA, shape_A, A, opB, shape_B, B, beta, shape_C, C, nullptr, ACTIVATION_LINEAR);
	}
	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		GemmRuntime rt = get_gemm_runtime(context, dtype, opA, opB, shape_A, shape_B);

		assert(shape_A.rank == 3 || shape_B.rank == 3);
		assert(shape_C.rank == 3);
		const int stride_A = size_of(dtype) * get_batch_stride(shape_A);
		const int stride_B = size_of(dtype) * get_batch_stride(shape_B);
		const int stride_C = size_of(dtype) * get_batch_stride(shape_C);
		const int stride_D = size_of(dtype) * get_batch_stride(shape_C);

		for (int i = 0; i < shape_C.dim[0]; i++)
		{
			rt.setMatrixA(getPointer<uint8_t>(A) + i * stride_A, shape_A, dtype, opA);
			rt.setMatrixB(getPointer<uint8_t>(B) + i * stride_B, shape_B, dtype, opB);
			rt.setMatrixC(getPointer<uint8_t>(C) + i * stride_C, shape_C, dtype);
			rt.setMatrixD(getPointer<uint8_t>(C) + i * stride_D, shape_C, dtype);
			if (i == 0)
			{
				rt.setScalingFactors(alpha, beta);
				rt.setup(context);
			}
			rt.run();
		}
	}

	void cpu_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias, mlActivationType_t act)
	{
		GemmRuntime rt = get_gemm_runtime(context, dtype, opA, opB, shape_A, shape_B);

		rt.setMatrixA(A, shape_A, dtype, opA);
		rt.setMatrixB(B, shape_B, dtype, opB);
		rt.setMatrixC(C, shape_C, dtype);
		rt.setMatrixD(D, shape_D, dtype);
		rt.setScalingFactors(alpha, beta);
		rt.useRelu(act == ACTIVATION_RELU);
		rt.setBias(bias, make_shape( { shape_D.dim[1] }), dtype);
		rt.setup(context);
		rt.run();

		if (act != ACTIVATION_RELU and act != ACTIVATION_LINEAR)
			cpu_activation_forward(context, dtype, shape_D, D, D, act);
	}

} /* namespace ml */

