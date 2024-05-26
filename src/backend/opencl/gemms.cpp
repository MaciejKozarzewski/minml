/*
 * gemms.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"

#include <CL/opencl.hpp>
#include <clblast.h>
#include <clblast_half.h>

#include <cassert>
#include <iostream>

namespace ml
{

	void opencl_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		assert(context != nullptr);

		clblast::Transpose op_A = is_transpose(opA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;
		clblast::Transpose op_B = is_transpose(opB) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

		const int M = is_transpose(opA) ? shape_A.dim[1] : shape_A.dim[0];
		const int N = is_transpose(opB) ? shape_B.dim[0] : shape_B.dim[1];
		const int K = is_transpose(opA) ? shape_A.dim[0] : shape_A.dim[1];

		const int LDA = get_last_dim(shape_A);
		const int LDB = get_last_dim(shape_B);
		const int LDC = get_last_dim(shape_C);

		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		const cl::Buffer &a_buffer = opencl::getMemoryObject(A).buffer();
		const cl::Buffer &b_buffer = opencl::getMemoryObject(B).buffer();
		cl::Buffer &c_buffer = opencl::getMemoryObject(C).buffer();
		cl::Event &event = *opencl::Context::getLastEvent(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				const cl_half _alpha = FloatToHalf(alpha);
				const cl_half _beta = FloatToHalf(beta);
				clblast::StatusCode status = clblast::Gemm(clblast::Layout::kRowMajor, op_A, op_B, M, N, K, _alpha, a_buffer(), 0, LDA, b_buffer(), 0,
						LDB, _beta, c_buffer(), 0, LDC, &queue(), &event());
				CHECK_OPENCL_STATUS(static_cast<cl_int>(status));
				break;
			}
			case DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				clblast::StatusCode status = clblast::Gemm(clblast::Layout::kRowMajor, op_A, op_B, M, N, K, _alpha, a_buffer(), 0, LDA, b_buffer(), 0,
						LDB, _beta, c_buffer(), 0, LDC, &queue(), &event());
				CHECK_OPENCL_STATUS(static_cast<cl_int>(status));
				break;
			}
			default:
				break;
		}
	}

	void opencl_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		assert(context != nullptr);
		assert(get_first_dim(shape_A) == get_first_dim(shape_B) && get_first_dim(shape_B) == get_first_dim(shape_C)); // uniform batch size

		const int batch = get_first_dim(shape_A);
		clblast::Transpose op_A = is_transpose(opA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;
		clblast::Transpose op_B = is_transpose(opB) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

		const int M = is_transpose(opA) ? shape_A.dim[2] : shape_A.dim[1];
		const int N = is_transpose(opB) ? shape_B.dim[1] : shape_B.dim[2];
		const int K = is_transpose(opA) ? shape_A.dim[1] : shape_A.dim[2];

		const int LDA = get_last_dim(shape_A);
		const int LDB = get_last_dim(shape_B);
		const int LDC = get_last_dim(shape_C);

		const int strideA = volume_without_first_dim(shape_A);
		const int strideB = volume_without_first_dim(shape_B);
		const int strideC = volume_without_first_dim(shape_C);

		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		const cl::Buffer &a_buffer = opencl::getMemoryObject(A).buffer();
		const cl::Buffer &b_buffer = opencl::getMemoryObject(B).buffer();
		cl::Buffer &c_buffer = opencl::getMemoryObject(C).buffer();
		cl::Event &event = *opencl::Context::getLastEvent(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				const cl_half _alpha = FloatToHalf(alpha);
				const cl_half _beta = FloatToHalf(beta);
				clblast::StatusCode status = clblast::GemmStridedBatched(clblast::Layout::kRowMajor, op_A, op_B, M, N, K, _alpha, a_buffer(), 0, LDA,
						strideA, b_buffer(), 0, LDB, strideB, _beta, c_buffer(), 0, LDC, strideC, batch, &queue(), &event());
				assert(status == clblast::StatusCode::kSuccess);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				clblast::StatusCode status = clblast::GemmStridedBatched(clblast::Layout::kRowMajor, op_A, op_B, M, N, K, _alpha, a_buffer(), 0, LDA,
						strideA, b_buffer(), 0, LDB, strideB, _beta, c_buffer(), 0, LDC, strideC, batch, &queue(), &event());
				assert(status == clblast::StatusCode::kSuccess);
				break;
			}
			default:
				break;
		}
	}
} /* namespace ml */

