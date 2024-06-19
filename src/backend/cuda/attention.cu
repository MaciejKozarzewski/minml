/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "helpers/indexers.cuh"
#include <cuda_runtime_api.h>

#include <cinttypes>

namespace
{
	using namespace ml;

	__device__ void* apply_offset(void *ptr, int offsetInBytes)
	{
		return reinterpret_cast<uint8_t*>(ptr) + offsetInBytes;
	}
	__global__ void kernel_calculate_pointers(void *q_ptr[], void *k_ptr[], void *v_ptr[], void *input, void *qk_ptr[], void *workspace,
			void *out_ptr[], void *output, int batch_size, int tokens, int num_heads, int head_dim, int dtype_size)
	{
		const Indexer<5> input_indexer(batch_size, tokens, 3, num_heads, head_dim);
		const Indexer<4> workspace_indexer(batch_size, num_heads, tokens, tokens);
		const Indexer<4> output_indexer(batch_size, tokens, num_heads, head_dim);

		for (int i = threadIdx.x; i < batch_size * num_heads; i += blockDim.x)
		{
			const int idx_b = i / num_heads;
			const int idx_h = i % num_heads;

			q_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 0, idx_h, 0));
			k_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 1, idx_h, 0));
			v_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 2, idx_h, 0));
			qk_ptr[i] = apply_offset(workspace, dtype_size * workspace_indexer.at(idx_b, idx_h, 0, 0));
			out_ptr[i] = apply_offset(output, dtype_size * output_indexer.at(idx_b, 0, idx_h, 0));
		}
	}

	void gemm_batched(mlContext_t context, char opA, char opB, mlDataType_t dtype, int M, int N, int K, float alpha, const void *A[], int lda,
			const void *B[], int ldb, float beta, void *C[], int ldc, int batch_count)
	{
		cublasOperation_t transa = is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transb = is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle = cuda::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (cuda::has_fp16_math(context))
				{
					const half _alpha = alpha;
					const half _beta = beta;
					cublasStatus_t status = cublasHgemmBatched(handle, transb, transa, N, M, K, &_alpha, getPointer<half*>(B), ldb,
							getPointer<half*>(A), lda, &_beta, getPointer<half*>(C), ldc, batch_count);
					assert(status == CUBLAS_STATUS_SUCCESS);
					break;
				}
				else
				{
					const float _alpha = alpha;
					const float _beta = beta;
					cublasStatus_t status = cublasGemmBatchedEx(handle, transb, transa, N, M, K, &_alpha, getPointer<void*>(B), CUDA_R_16F, ldb,
							getPointer<void*>(A), CUDA_R_16F, lda, &_beta, getPointer<void*>(C), CUDA_R_16F, ldc, batch_count, CUBLAS_COMPUTE_32F,
							CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
					break;
				}
			}
			case DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				cublasStatus_t status = cublasSgemmBatched(handle, transb, transa, N, M, K, &_alpha, getPointer<float*>(B), ldb,
						getPointer<float*>(A), lda, &_beta, getPointer<float*>(C), ldc, batch_count);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}
}

namespace ml
{
	int cuda_multi_head_attention_get_workspace_size(mlShape_t shape, int num_heads, bool training)
	{
		assert(shape.rank == 3);
		const int batch_size = shape.dim[0];
		const int tokens = shape.dim[1];

		int result = batch_size * num_heads * tokens * tokens;
		if (training)
			result *= 2;
		return result;
	}
	void cuda_multi_head_attention_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, int num_heads,
			void *workspace)
	{
		assert(shape.rank == 3);
		const int batch_size = shape.dim[0];
		const int tokens = shape.dim[1];
		const int embedding = shape.dim[2] / 3;
		const int head_dim = embedding / num_heads;

		const int num_pointers = batch_size * num_heads;
		void **pointers = getPointer<void*>(cuda::Context::getWorkspace(context));

		void **q_ptr = pointers + 0 * num_pointers;
		void **k_ptr = pointers + 1 * num_pointers;
		void **v_ptr = pointers + 2 * num_pointers;
		void **qk_ptr = pointers + 3 * num_pointers;
		void **out_ptr = pointers + 4 * num_pointers;

		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(q_ptr, k_ptr, v_ptr, const_cast<void*>(input), qk_ptr, workspace, out_ptr, output,
				batch_size, tokens, num_heads, head_dim, size_of(dtype));

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', dtype, tokens, tokens, head_dim, scale, const_cast<const void**>(q_ptr), 3 * embedding,
				const_cast<const void**>(k_ptr), 3 * embedding, 0.0f, qk_ptr, tokens, num_pointers);

		const mlShape_t qk_shape = make_shape( { batch_size * num_heads * tokens, tokens });
		cuda_activation_forward(context, dtype, qk_shape, workspace, workspace, ACTIVATION_SOFTMAX);

		gemm_batched(context, 'n', 'n', dtype, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(v_ptr), 3 * embedding, 0.0f, out_ptr, embedding, num_pointers);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_multi_head_attention_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
			int num_heads, void *workspace)
	{
		assert(shape.rank == 3);
		const int batch_size = shape.dim[0];
		const int tokens = shape.dim[1];
		const int embedding = shape.dim[2] / 3;
		const int head_dim = embedding / num_heads;

		void *forward_workspace = workspace;
		void *backward_workspace = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace)
				+ batch_size * num_heads * tokens * tokens * size_of(DTYPE_FLOAT32));

		const int num_pointers = batch_size * num_heads;
		void **pointers = getPointer<void*>(cuda::Context::getWorkspace(context));

		void **q_ptr = pointers + 0 * num_pointers;
		void **k_ptr = pointers + 1 * num_pointers;
		void **v_ptr = pointers + 2 * num_pointers;
		void **qk_ptr = pointers + 3 * num_pointers;
		void **out_ptr = pointers + 4 * num_pointers;

		void **dq_ptr = pointers + 5 * num_pointers;
		void **dk_ptr = pointers + 6 * num_pointers;
		void **dv_ptr = pointers + 7 * num_pointers;
		void **dqk_ptr = pointers + 8 * num_pointers;
		void **dout_ptr = pointers + 9 * num_pointers;

		cudaStream_t stream = cuda::Context::getStream(context);

		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(q_ptr, k_ptr, v_ptr, const_cast<void*>(input), qk_ptr, forward_workspace, out_ptr, nullptr,
				batch_size, tokens, num_heads, head_dim, size_of(DTYPE_FLOAT32));
		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(dq_ptr, dk_ptr, dv_ptr, gradient_prev, dqk_ptr, backward_workspace, dout_ptr, gradient_next,
				batch_size, tokens, num_heads, head_dim, size_of(DTYPE_FLOAT32));

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, scale, const_cast<const void**>(q_ptr), 3 * embedding,
				const_cast<const void**>(k_ptr), 3 * embedding, 0.0f, qk_ptr, tokens, num_pointers);

		const mlShape_t qk_shape = make_shape( { batch_size * num_heads * tokens, tokens });
		cuda_activation_forward(context, DTYPE_FLOAT32, qk_shape, forward_workspace, forward_workspace, ACTIVATION_SOFTMAX);

		// dqk = dy * V^T
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, 1.0f, const_cast<const void**>(dout_ptr), embedding,
				const_cast<const void**>(v_ptr), 3 * embedding, 0.0f, dqk_ptr, tokens, num_pointers);
		// dV = qk^T * dy
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(dout_ptr), embedding, 0.0f, dv_ptr, 3 * embedding, num_pointers);

		cuda_activation_backward(context, qk_shape, backward_workspace, backward_workspace, forward_workspace, ACTIVATION_SIGMOID); // softmax and sigmoid have the same backward equations

		// dQ = dqk * K
		gemm_batched(context, 'n', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(k_ptr), 3 * embedding, 0.0f, dq_ptr, 3 * embedding, num_pointers);
		// dK = dqk^T * Q
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(q_ptr), 3 * embedding, 0.0f, dk_ptr, 3 * embedding, num_pointers);

		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

