/*
 * group_conv.cu
 *
 *  Created on: Mar 21, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "activations.cuh"
#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../helpers/misc.cuh"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;
	using namespace ml;

	template<typename T>
	__global__ void kernel_channel_shuffle(float beta, T *output, const T *input, int first_dim, int last_dim, int groups, bool invert)
	{
		assert(last_dim % groups == 0);
		extern __shared__ char shared_array[];
		T *workspace = reinterpret_cast<T*>(shared_array); // blockDim.y * last_dim
		int16_t *shuffled_indices = reinterpret_cast<int16_t*>(workspace + blockDim.y * last_dim); // last_dim

		const Indexer<2> group_indexer(groups, last_dim / groups);
		for (int j = threadIdx.y * blockDim.x + threadIdx.x; j < last_dim; j += blockDim.x * blockDim.y)
		{
			const int src_idx = j;
			const int dst_idx = group_indexer.at(j % groups, j / groups);
			if (invert)
				shuffled_indices[dst_idx] = src_idx;
			else
				shuffled_indices[src_idx] = dst_idx;
		}
		__syncthreads();

		const Indexer<2> indexer(first_dim, last_dim);

		for (int i = blockIdx.x * blockDim.y; i < first_dim; i += blockDim.y * gridDim.x)
		{
			const int tokens_left = min(blockDim.y, first_dim - i);
			const int offset = indexer.at(i, 0);
			if ((tokens_left * last_dim) % 4 == 0 && !std::is_same<T, double>::value)
			{
				for (int j = 4 * threadIdx.x; j < tokens_left * last_dim; j += blockDim.x * 4)
					vectors::vector_copy<4>(workspace + j, input + offset + j);
			}
			else
			{
				for (int j = threadIdx.x; j < tokens_left * last_dim; j += blockDim.x)
					vectors::vector_copy<1>(workspace + j, input + offset + j);
			}
			__syncthreads();

			if (i + threadIdx.y < first_dim)
			{
				for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
				{
					T tmp = workspace[threadIdx.y * last_dim + shuffled_indices[j]];
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
					if (beta != 0.0f)
						tmp += static_cast<T>(beta) * output[indexer.at(i + threadIdx.y, j)];
#endif
					output[indexer.at(i + threadIdx.y, j)] = tmp;
				}
			}
			__syncthreads();
		}
	}

}

namespace ml
{
	void cuda_channel_shuffle(mlContext_t context, const mlTensor_t x, float beta, mlTensor_t y, int groups, bool invert)
	{
		assert(same_shape(x, y));
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(last_dim % groups == 0);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(32, 8);
		dim3 gridDim(std::min(1024, first_dim));

		const int shared_memory_size = blockDim.y * last_dim * size_of(x.dtype) + last_dim * sizeof(int16_t);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_channel_shuffle<<<gridDim, blockDim, shared_memory_size, stream >>>(beta, data<half>(y), data<half>(x), first_dim, last_dim,
						groups, invert);
				break;
			case DTYPE_FLOAT32:
				kernel_channel_shuffle<<<gridDim, blockDim, shared_memory_size, stream >>>(beta, data<float>(y), data<float>(x), first_dim, last_dim,
						groups, invert);
				break;
			case DTYPE_FLOAT64:
				kernel_channel_shuffle<<<gridDim, blockDim, shared_memory_size, stream >>>(beta, data<double>(y), data<double>(x), first_dim,
						last_dim, groups, invert);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_group_gemm(mlContext_t context, char opA, char opB, float alpha, const mlTensor_t A, const mlTensor_t B, float beta, mlTensor_t C,
			int groups)
	{
		assert(A.rank == 2);
		assert(B.rank == 2);
		assert(C.rank == 2);
		assert(A.dim[1] % groups == 0);
		assert(B.dim[1] % groups == 0);
		assert(C.dim[1] % groups == 0);
		cublasOperation_t op_A = is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_B = is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		const int M = is_transpose(opB) ? B.dim[0] : (B.dim[1] / groups);
		const int N = is_transpose(opA) ? (A.dim[1] / groups) : A.dim[0];
		const int K = is_transpose(opB) ? (B.dim[1] / groups) : B.dim[0];

		const int LDA = A.dim[1];
		const int LDB = B.dim[1];
		const int LDC = C.dim[1];
		const int strideA = A.dim[1] / groups;
		const int strideB = A.dim[1] / groups;
		const int strideC = C.dim[1] / groups;

		cublasHandle_t handle = ml::cuda_backend::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (C.dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(is_fp16(A));
				assert(is_fp16(B));
				const half _alpha = alpha;
				const half _beta = beta;
				cublasStatus_t status = cublasHgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<half>(B), LDB, strideB, data<half>(A),
						LDA, strideA, &_beta, data<half>(C), LDC, strideC, groups);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(A.dtype == B.dtype);
				assert(is_fp32(A) || is_fp16(A));
				const float _alpha = alpha;
				const float _beta = beta;
				if (is_fp32(A))
				{
					if (ml::cuda_backend::Context::allowsTF32(context))
					{
						cublasStatus_t status = cublasGemmStridedBatchedEx(handle, op_B, op_A, M, N, K, &_alpha, B.data, CUDA_R_32F, LDB, strideB,
								A.data, CUDA_R_32F, LDA, strideA, &_beta, C.data, CUDA_R_32F, LDC, strideC, groups, CUBLAS_COMPUTE_32F_FAST_TF32,
								CUBLAS_GEMM_DEFAULT);
						assert(status == CUBLAS_STATUS_SUCCESS);
					}
					else
					{
						cublasStatus_t status = cublasSgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<float>(B), LDB, strideB,
								data<float>(A), LDA, strideA, &_beta, data<float>(C), LDC, strideC, groups);
						assert(status == CUBLAS_STATUS_SUCCESS);
					}
				}
				else
				{
					cublasStatus_t status = cublasGemmStridedBatchedEx(handle, op_B, op_A, M, N, K, &_alpha, B.data, CUDA_R_16F, LDB, strideB, A.data,
							CUDA_R_16F, LDA, strideA, &_beta, C.data, CUDA_R_32F, LDC, strideC, groups, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				break;
			}
			case DTYPE_FLOAT64:
			{
				const double _alpha = alpha;
				const double _beta = beta;
				cublasStatus_t status = cublasDgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<double>(B), LDB, strideB,
						data<double>(A), LDA, strideA, &_beta, data<double>(C), LDC, strideC, groups);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}

} /* namespace ml */

