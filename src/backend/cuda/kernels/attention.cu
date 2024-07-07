/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cinttypes>
#include <iostream>

namespace
{
	template<typename T>
	__device__ T clamp(T x, T lower, T upper)
	{
		assert(lower <= upper);
		return max(lower, min(upper, x));
	}
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

	template<typename T>
	__global__ void kernel_softmax_forward_in_place(T *input, const T *weights, int batch_size, int num_heads, int height, int width, int range)
	{
		assert(height * width <= 512);
		assert(blockDim.x == 128);
		__shared__ float workspace[512];
		__shared__ int2 indices[512];
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<128> tile = cg::tiled_partition<128>(thb);

		const int batch_idx = blockIdx.y;
		const int head_idx = blockIdx.z;
		const int tokens = height * width;

		const Indexer<3> weight_indexer(num_heads, 2 * range + 1, 2 * range + 1);
		const Indexer<4> input_indexer(batch_size, num_heads, tokens, tokens);

		for (int i = threadIdx.x; i < height * width; i += blockDim.x)
		{
			indices[i].x = i / width;
			indices[i].y = i % width;
		}
		__syncthreads();

		for (int i = blockIdx.x; i < tokens; i += gridDim.x)
		{
			const int2 origin = indices[i];

			const int idx = input_indexer.at(batch_idx, head_idx, i, 0);
			float max_value = -1e+32f;
			for (int j = tile.thread_rank(); j < tokens; j += tile.size())
			{
				const int2 current = indices[j];
				const int offset_x = range + clamp(current.x - origin.x, -range, range);
				const int offset_y = range + clamp(current.y - origin.y, -range, range);
				const float bias = static_cast<float>(weights[weight_indexer.at(head_idx, offset_x, offset_y)]);
				workspace[j] = bias + static_cast<float>(input[idx + j]);
				max_value = max(max_value, workspace[j]);
			}
			const float shift = cg::reduce(tile, max_value, cg::greater<float>());

			float partial_sum = 0.0f;
			for (int j = tile.thread_rank(); j < tokens; j += tile.size())
			{
				workspace[j] = exp(workspace[j] - shift);
				partial_sum += workspace[j];
			}
			const float inv_sum = 1.0f / cg::reduce(tile, partial_sum, cg::plus<float>());
			for (int j = tile.thread_rank(); j < tokens; j += tile.size())
				input[idx + j] = static_cast<T>(workspace[j] * inv_sum);
		}
	}
	__global__ void kernel_softmax_backward_in_place(const float *output, float *gradient, float *weights_update, int batch_size, int num_heads,
			int height, int width, int range)
	{
		assert(height * width <= 512);
		__shared__ int2 indices[512];
		__shared__ float workspace[2048];

		const int batch_idx = blockIdx.x;
		const int head_idx = blockIdx.y;
		const int tokens = height * width;
		const int size = 2 * range + 1;

		for (int i = threadIdx.x; i < tokens; i += blockDim.x)
		{
			indices[i].x = i / width;
			indices[i].y = i % width;
		}
		assert(size * size <= 2048);
		for (int i = threadIdx.x; i < size * size; i += blockDim.x)
			workspace[i] = 0.0f;
		__syncthreads();

		const Indexer<4> gradient_indexer(batch_size, num_heads, tokens, tokens);
		for (int i = 0; i < tokens; i++)
		{
			const int2 origin = indices[i];

			const int idx = gradient_indexer.at(batch_idx, head_idx, i, 0);
			for (int j = threadIdx.x; j < tokens; j += blockDim.x)
			{
				const int2 current = indices[j];
				const int offset_h = range + clamp(current.x - origin.x, -range, range);
				const int offset_w = range + clamp(current.y - origin.y, -range, range);
				const float out = output[idx + j];
				const float grad = gradient[idx + j] * out * (1.0f - out);

				atomicAdd(workspace + offset_h * size + offset_w, grad);
				gradient[idx + j] = grad;
			}
			__syncthreads();
		}

		const Indexer<3> update_indexer(batch_size, num_heads, size * size);
		for (int i = threadIdx.x; i < size * size; i += blockDim.x)
			weights_update[update_indexer.at(batch_idx, head_idx, i)] += workspace[i];
	}
	__global__ void kernel_weights_update_reduction(const float *workspace, float *update, int batch_size, int num_heads, int last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ float storage[32 * 32];

		const Indexer<3> workspace_indexer(batch_size, num_heads, last_dim);

		const int tid = blockIdx.x * 32 + threadIdx.x;
		float d_w = 0.0f;
		if (tid < last_dim)
			for (int i = threadIdx.y; i < batch_size; i += 32)
				d_w += workspace[workspace_indexer.at(i, blockIdx.y, tid)];
		storage[threadIdx.y * 32 + threadIdx.x] = d_w;

		__syncthreads();
		assert(blockDim.x == 32 && blockDim.y == 32);
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				storage[threadIdx.y * 32 + threadIdx.x] += storage[(i + threadIdx.y) * 32 + threadIdx.x];
			__syncthreads();
		}

		const Indexer<2> update_indexer(num_heads, last_dim);
		if (threadIdx.y == 0 && tid < last_dim)
			update[update_indexer.at(blockIdx.y, tid)] += storage[threadIdx.x];
	}

	void gemm_batched(ml::mlContext_t context, char opA, char opB, ml::mlDataType_t dtype, int M, int N, int K, float alpha, const void *A[], int lda,
			const void *B[], int ldb, float beta, void *C[], int ldc, int batch_count)
	{
		cublasOperation_t transa = ml::is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transb = ml::is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle = ml::cuda::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
			{
				if (ml::cuda::has_fp16_math(context))
				{
					const half _alpha = alpha;
					const half _beta = beta;
					cublasStatus_t status = cublasHgemmBatched(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<half*>(B), ldb,
							ml::getPointer<half*>(A), lda, &_beta, ml::getPointer<half*>(C), ldc, batch_count);
					assert(status == CUBLAS_STATUS_SUCCESS);
					break;
				}
				else
				{
					const float _alpha = alpha;
					const float _beta = beta;
					cublasStatus_t status = cublasGemmBatchedEx(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<void*>(B), CUDA_R_16F, ldb,
							ml::getPointer<void*>(A), CUDA_R_16F, lda, &_beta, ml::getPointer<void*>(C), CUDA_R_16F, ldc, batch_count,
							CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
					break;
				}
			}
			case ml::DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				cublasStatus_t status = cublasSgemmBatched(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<float*>(B), ldb,
						ml::getPointer<float*>(A), lda, &_beta, ml::getPointer<float*>(C), ldc, batch_count);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}
	void run_softmax_forward(cudaStream_t stream, void *input, ml::mlShape_t input_shape, const void *weights, ml::mlShape_t weights_shape,
			ml::mlDataType_t dtype)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int num_heads = weights_shape.dim[0];
		assert(weights_shape.dim[1] == weights_shape.dim[2]);
		const int range = (weights_shape.dim[1] - 1) / 2;

		dim3 blockDim(128);
		dim3 gridDim(32, batch_size, num_heads);

		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
			{
				kernel_softmax_forward_in_place<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<half>(input), ml::getPointer<half>(weights),
						batch_size, num_heads, height, width, range);
				break;
			}
			case ml::DTYPE_FLOAT32:
			{
				kernel_softmax_forward_in_place<<<gridDim, blockDim, 0, stream>>>(ml::getPointer<float>(input), ml::getPointer<float>(weights),
						batch_size, num_heads, height, width, range);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	int cuda_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, bool training)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1] * input_shape.dim[2];
		const int num_heads = weights_shape.dim[0];

		int result = batch_size * num_heads * tokens * tokens;
		if (training)
			result = result * 2 + batch_size * num_heads * weights_shape.dim[1] * weights_shape.dim[2];
		return result;
	}
	void cuda_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype, const void *input,
			void *output, const void *weights, void *workspace)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / 3;
		const int num_heads = weights_shape.dim[0];
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
		assert(cudaGetLastError() == cudaSuccess);

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', dtype, tokens, tokens, head_dim, scale, const_cast<const void**>(q_ptr), 3 * embedding,
				const_cast<const void**>(k_ptr), 3 * embedding, 0.0f, qk_ptr, tokens, num_pointers);

		run_softmax_forward(stream, workspace, input_shape, weights, weights_shape, dtype);

		gemm_batched(context, 'n', 'n', dtype, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(v_ptr), 3 * embedding, 0.0f, out_ptr, embedding, num_pointers);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, void *gradient_prev, void *gradient_next, void *weights_update, void *workspace)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / 3;
		const int num_heads = weights_shape.dim[0];
		const int head_dim = embedding / num_heads;
		assert(weights_shape.dim[1] == weights_shape.dim[2]);
		const int range = (weights_shape.dim[1] - 1) / 2;

		const int offset = batch_size * num_heads * tokens * tokens * size_of(DTYPE_FLOAT32);
		void *forward_workspace = workspace;
		void *backward_workspace = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace) + offset);
		void *update_workspace = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace) + 2 * offset);

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

		run_softmax_forward(stream, forward_workspace, input_shape, weights, weights_shape, DTYPE_FLOAT32);

		// dqk = dy * V^T
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, 1.0f, const_cast<const void**>(dout_ptr), embedding,
				const_cast<const void**>(v_ptr), 3 * embedding, 0.0f, dqk_ptr, tokens, num_pointers);
		// dV = qk^T * dy
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(dout_ptr), embedding, 0.0f, dv_ptr, 3 * embedding, num_pointers);

		dim3 blockDim(256);
		dim3 gridDim(batch_size, num_heads);
		kernel_softmax_backward_in_place<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(forward_workspace),
				getPointer<float>(backward_workspace), getPointer<float>(update_workspace), batch_size, num_heads, height, width, range);
		assert(cudaGetLastError() == cudaSuccess);

		const int last_dim = weights_shape.dim[1] * weights_shape.dim[2];
		blockDim = dim3(32, 32);
		gridDim = dim3((last_dim + 31) / 32, num_heads);
		kernel_weights_update_reduction<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(update_workspace), getPointer<float>(weights_update),
				batch_size, num_heads, last_dim);
		assert(cudaGetLastError() == cudaSuccess);

// dQ = dqk * K
		gemm_batched(context, 'n', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f / scale, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(k_ptr), 3 * embedding, 0.0f, dq_ptr, 3 * embedding, num_pointers);
		// dK = dqk^T * Q
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f / scale, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(q_ptr), 3 * embedding, 0.0f, dk_ptr, 3 * embedding, num_pointers);

		assert(cudaGetLastError() == cudaSuccess);
	}
} /* namespace ml */

