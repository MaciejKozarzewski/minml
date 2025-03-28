/*
 * gemms.cpp
 *
 *  Created on: Jan 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>

namespace
{
	using namespace ml;
	int get_batch_stride(const mlShape_t &shape) noexcept
	{
		switch (shape.rank)
		{
			default:
			case 2:
				return 0;
			case 3:
				return shape.dim[1] * shape.dim[2];
		}
	}
	int num_rows(const mlShape_t &shape) noexcept
	{
		switch (shape.rank)
		{
			default:
				return 0;
			case 2:
				return shape.dim[0];
			case 3:
				return shape.dim[1];
		}
	}
	int num_columns(const mlShape_t &shape) noexcept
	{
		switch (shape.rank)
		{
			default:
				return 0;
			case 2:
				return shape.dim[1];
			case 3:
				return shape.dim[2];
		}
	}
}

namespace ml
{

	void cuda_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		assert(context != nullptr);
		cublasOperation_t op_A = is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_B = is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		const int M = is_transpose(opB) ? num_rows(shape_B) : num_columns(shape_B);
		const int N = is_transpose(opA) ? num_columns(shape_A) : num_rows(shape_A);
		const int K = is_transpose(opB) ? num_columns(shape_B) : num_rows(shape_B);

		const int LDA = get_last_dim(shape_A);
		const int LDB = get_last_dim(shape_B);
		const int LDC = get_last_dim(shape_C);

		cublasHandle_t handle = cuda::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (dtype)
		{
			case DTYPE_INT32: // AB [int8], C[int32]
			{
				assert(K % 4 == 0);
				const int32_t _alpha = static_cast<int32_t>(alpha);
				const int32_t _beta = static_cast<int32_t>(beta);
				cublasStatus_t status = cublasGemmEx(handle, op_B, op_A, M, N, K, &_alpha, getPointer<void>(B), CUDA_R_8I, LDB, getPointer<void>(A),
						CUDA_R_8I, LDA, &_beta, getPointer<void>(C), CUDA_R_32I, LDC, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case DTYPE_FLOAT16: // ABC [float16]
			{
				const half _alpha = alpha;
				const half _beta = beta;
				cublasStatus_t status = cublasHgemm(handle, op_B, op_A, M, N, K, &_alpha, getPointer<half>(B), LDB, getPointer<half>(A), LDA, &_beta,
						getPointer<half>(C), LDC);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case DTYPE_FLOAT32: // ABC [float32]
			{
				const float _alpha = alpha;
				const float _beta = beta;
				if (ml::cuda::Context::allowsTF32(context))
				{
					cublasStatus_t status = cublasGemmEx(handle, op_B, op_A, M, N, K, &_alpha, getPointer<void>(B), CUDA_R_32F, LDB,
							getPointer<void>(A), CUDA_R_32F, LDA, &_beta, getPointer<void>(C), CUDA_R_32F, LDC, CUBLAS_COMPUTE_32F_FAST_TF32,
							CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				else
				{
					cublasStatus_t status = cublasSgemm(handle, op_B, op_A, M, N, K, &_alpha, getPointer<float>(B), LDB, getPointer<float>(A), LDA,
							&_beta, getPointer<float>(C), LDC);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				break;
			}
			case DTYPE_FLOAT64: // ABC [float64]
			{
				const double _alpha = alpha;
				const double _beta = beta;
				cublasStatus_t status = cublasDgemm(handle, op_B, op_A, M, N, K, &_alpha, getPointer<double>(B), LDB, getPointer<double>(A), LDA,
						&_beta, getPointer<double>(C), LDC);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}

	void cuda_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		assert(context != nullptr);
		assert(shape_A.rank == 3 || shape_B.rank == 3);
		assert(shape_C.rank == 3);
		const int batch = get_first_dim(shape_C);
		cublasOperation_t op_A = is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_B = is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		const int M = is_transpose(opB) ? num_rows(shape_B) : num_columns(shape_B);
		const int N = is_transpose(opA) ? num_columns(shape_A) : num_rows(shape_A);
		const int K = is_transpose(opB) ? num_columns(shape_B) : num_rows(shape_B);

		const int LDA = get_last_dim(shape_A);
		const int LDB = get_last_dim(shape_B);
		const int LDC = get_last_dim(shape_C);
		const int strideA = get_batch_stride(shape_A);
		const int strideB = get_batch_stride(shape_B);
		const int strideC = get_batch_stride(shape_C);

		cublasHandle_t handle = cuda::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				const half _alpha = alpha;
				const half _beta = beta;
				cublasStatus_t status = cublasHgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, getPointer<half>(B), LDB, strideB,
						getPointer<half>(A), LDA, strideA, &_beta, getPointer<half>(C), LDC, strideC, batch);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				if (ml::cuda::Context::allowsTF32(context))
				{
					cublasStatus_t status = cublasGemmStridedBatchedEx(handle, op_B, op_A, M, N, K, &_alpha, getPointer<void>(B), CUDA_R_32F, LDB,
							strideB, getPointer<void>(A), CUDA_R_32F, LDA, strideA, &_beta, getPointer<void>(C), CUDA_R_32F, LDC, strideC, batch,
							CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				else
				{
					cublasStatus_t status = cublasSgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, getPointer<float>(B), LDB, strideB,
							getPointer<float>(A), LDA, strideA, &_beta, getPointer<float>(C), LDC, strideC, batch);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				break;
			}
			case DTYPE_FLOAT64:
			{
				const double _alpha = alpha;
				const double _beta = beta;
				cublasStatus_t status = cublasDgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, getPointer<double>(B), LDB, strideB,
						getPointer<double>(A), LDA, strideA, &_beta, getPointer<double>(C), LDC, strideC, batch);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}
} /* namespace ml */

