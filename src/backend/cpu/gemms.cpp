/*
 * gemms.cpp
 *
 *  Created on: Sep 5, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#ifdef USE_OPENBLAS
#  ifdef __linux__
#    include <cblas.h>
#  else
#    include <openblas/cblas.h>
#  endif
#endif

#include <cassert>

namespace
{
	using namespace ml;

#ifdef USE_OPENBLAS
	bool is_transpose(char c) noexcept
	{
		assert(c == 'T' || c == 't' || c == 'N' || c == 'n');
		return c == 'T' || c == 't';
	}

	void openblas_gemm(mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B, const void *B, char opA,
			char opB, float alpha, float beta)
	{
		CBLAS_TRANSPOSE op_A = is_transpose(opA) ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE op_B = is_transpose(opB) ? CblasTrans : CblasNoTrans;

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
#ifdef USE_OPENBLAS
				cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, alpha, getPointer<float>(A), LDA, getPointer<float>(B), LDB, beta,
						getPointer<float>(C), LDC);
#endif
				break;
			}
			default:
				assert(false);
		}
	}
	void openblas_gemm_batched(mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B, const void *B,
			char opA, char opB, float alpha, float beta)
	{
		CBLAS_TRANSPOSE op_A = is_transpose(opA) ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE op_B = is_transpose(opB) ? CblasTrans : CblasNoTrans;

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
#ifdef USE_OPENBLAS
					cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, alpha, A_ptr, LDA, B_ptr, LDB, beta, C_ptr, LDC);
#endif
				}
				break;
			}
			default:
				assert(false);
		}
	}
#endif
}

namespace ml
{

	void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
#ifdef USE_OPENBLAS
		openblas_gemm(dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
#else
		assert(false);
#endif
	}
	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
#ifdef USE_OPENBLAS
		openblas_gemm_batched(dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
#else
		assert(false);
#endif
	}
} /* namespace ml */

