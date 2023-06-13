/*
 * openblas_gemm.cpp
 *
 *  Created on: Sep 5, 2020
 *      Author: Maciej Kozarzewski
 */

#ifdef USE_OPENBLAS

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#ifdef __linux__
#  include <cblas.h>
#else
#  include <openblas/cblas.h>
#endif

#include <cassert>

namespace ml
{

	void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		const CBLAS_TRANSPOSE op_A = is_transpose(opA) ? CblasTrans : CblasNoTrans;
		const CBLAS_TRANSPOSE op_B = is_transpose(opB) ? CblasTrans : CblasNoTrans;

		const int M = is_transpose(opA) ? shape_A.dim[1] : shape_A.dim[0];
		const int N = is_transpose(opB) ? shape_B.dim[0] : shape_B.dim[1];
		const int K = is_transpose(opA) ? shape_A.dim[0] : shape_A.dim[1];

		const int LDA = shape_A.dim[1];
		const int LDB = shape_B.dim[1];
		const int LDC = shape_C.dim[1];

		switch (dtype)
		{
			case DTYPE_FLOAT32:
			{
				cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, alpha, getPointer<float>(A), LDA, getPointer<float>(B), LDB, beta,
						getPointer<float>(C), LDC);
				break;
			}
			default:
				break;
		}
	}
	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		const CBLAS_TRANSPOSE op_A = is_transpose(opA) ? CblasTrans : CblasNoTrans;
		const CBLAS_TRANSPOSE op_B = is_transpose(opB) ? CblasTrans : CblasNoTrans;

		const int M = is_transpose(opA) ? shape_A.dim[2] : shape_A.dim[1];
		const int N = is_transpose(opB) ? shape_B.dim[1] : shape_B.dim[2];
		const int K = is_transpose(opA) ? shape_A.dim[1] : shape_A.dim[2];

		const int LDA = shape_A.dim[2];
		const int LDB = shape_B.dim[2];
		const int LDC = shape_C.dim[2];

		const int strideA = volume_without_first_dim(shape_A);
		const int strideB = volume_without_first_dim(shape_B);
		const int strideC = volume_without_first_dim(shape_C);

		const int batch = get_first_dim(shape_A);

		switch (dtype)
		{
			case DTYPE_FLOAT32:
			{
				for (int i = 0; i < batch; i++)
				{
					const float *A_ptr = getPointer<float>(A) + i * strideA;
					const float *B_ptr = getPointer<float>(B) + i * strideB;
					float *C_ptr = getPointer<float>(C) + i * strideC;
					cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, alpha, A_ptr, LDA, B_ptr, LDB, beta, C_ptr, LDC);
				}
				break;
			}
			default:
				break;
		}
	}
} /* namespace ml */

#endif

